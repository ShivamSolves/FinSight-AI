"""
screenshot_extractor.py — Turns a payment screenshot into clean transactions.

Where the CSV and PDF extractors read *structured* tables, this one reads an
*image* — a UPI / GPay / Paytm / bank confirmation screen — and asks a vision
LLM (Gemini) to pull out the transaction fields. The output converges on the
exact same schema as the other extractors, so everything downstream
(categorization, vector store, RAG) is reused unchanged:

    date | description | amount | transaction_type

Design
──────
The vision call and the parsing are deliberately split:

    extract_transactions_from_image(bytes)   ← does the network call
        └─ _call_vision(bytes) -> raw text    ← Gemini, isolated & mockable
        └─ parse_transaction_json(text)       ← PURE, unit-tested without net
        └─ transactions_from_records(list)    ← PURE, builds the raw DataFrame
        └─ clean_transactions(df)             ← shared with CSV/PDF

Prompting the model for strict JSON (response_mime_type="application/json")
makes parsing reliable; parse_transaction_json still tolerates code fences and
a single object, because models occasionally wrap or unwrap the payload.

Sign convention matches the rest of the app: money OUT (a payment) is negative,
money IN (a credit/refund) is positive. The model returns an explicit `type`
("debit"/"credit") so the sign is never guessed from context.

Public API
──────────
    extract_transactions_from_image(image_bytes, mime_type="image/png") -> DataFrame
    extract_transactions_from_file(filepath) -> DataFrame
    parse_transaction_json(text) -> list[dict]     (pure)
    transactions_from_records(records) -> DataFrame (pure)
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import pandas as pd

from finsight.config import LLM_MODEL
from finsight.ingestion.csv_extractor import clean_transactions

logger = logging.getLogger(__name__)

# Bounded retry for transient Gemini overload spikes (429/503).
_VISION_RETRIES = 3
_VISION_BACKOFF_SECONDS = 2


class ScreenshotExtractionError(RuntimeError):
    """Raised when a screenshot can't be turned into transactions.

    Distinct from a generic Exception so the UI can catch exactly this and show
    a friendly "couldn't read that screenshot" message (e.g. no API key, model
    returned nothing usable) instead of a traceback.
    """


# ── Prompt ────────────────────────────────────────────────────────────────────

_PROMPT = """You are extracting a UPI / bank payment from a confirmation screenshot.

These screenshots typically show: the amount, the receiver's UPI ID
(e.g. "swiggy@paytm"), a transaction ID, and a UTR / reference number. A payee
display name and a date/time may or may not be present.

Return ONLY a JSON array. Each element is one payment object:
{
  "amount": 123.45,               // positive number only — no currency symbols or commas
  "upi_id": "receiver@bank",      // the receiver's UPI/VPA exactly as shown; "" if absent
  "payee_name": "Name shown",     // the receiver display name if shown; "" if absent
  "utr": "reference number",      // UTR / bank reference / transaction id; "" if absent
  "date": "YYYY-MM-DD",           // the date shown; "" if no date is visible
  "type": "debit" | "credit"      // "debit" = money you paid out, "credit" = money you received
}

Rules:
- A confirmation screen is almost always ONE payment; return a single-element array.
- Only return multiple objects if the screenshot genuinely lists multiple payments.
- Do NOT invent a date, UPI ID, or UTR that isn't visible — use "" instead.
- Do NOT treat a balance, fee, or the UTR itself as a separate transaction.
- If there is no clear amount, return an empty array: []
- Output nothing except the JSON array.
"""


# ── Pure parsing (unit-testable, no network) ──────────────────────────────────

def parse_transaction_json(text: str) -> list[dict]:
    """
    Parse the model's raw output into a list of transaction dicts.

    Tolerates the shapes models actually return: a bare JSON array, an array
    wrapped in ```json fences, or a single object instead of an array. Returns
    [] (not an exception) when nothing usable is present, so callers decide how
    to react.
    """
    if not text or not text.strip():
        return []

    # Strip markdown code fences if present.
    cleaned = re.sub(r"```(?:json)?", "", text.strip()).strip().strip("`").strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fall back to the first balanced JSON array/object in the text.
        match = re.search(r"(\[.*\]|\{.*\})", cleaned, re.DOTALL)
        if not match:
            logger.warning("Vision model returned no parseable JSON.")
            return []
        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError:
            logger.warning("Vision model returned malformed JSON.")
            return []

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []
    return [r for r in data if isinstance(r, dict)]


def _build_description(record: dict) -> str:
    """
    Choose the best merchant/description text for categorization + display.

    A UPI screen rarely has a clean merchant name, so we fall back through:
      payee display name → the local part of the UPI ID ("swiggy@paytm" →
      "swiggy") → the full UPI ID. The local part matters most: it's the token
      the rule-based categorizer substring-matches against ("swiggy" → Food).
    """
    payee = str(record.get("payee_name", "")).strip()
    if payee:
        return payee

    upi_id = str(record.get("upi_id", "")).strip()
    if upi_id:
        return upi_id.split("@")[0].strip() or upi_id

    # Last resort: some models still return a plain description.
    return str(record.get("description", "")).strip()


def transactions_from_records(records: list[dict]) -> pd.DataFrame:
    """
    Convert parsed records into a raw DataFrame for clean_transactions().

    Applies the sign convention (debit → negative), derives a description from
    the payee/UPI ID, defaults a missing date to today, and de-duplicates on the
    UTR so re-reading the same confirmation can't double-count. Records missing
    an amount or any payee signal are skipped.
    """
    from datetime import date as _date

    today = _date.today().isoformat()
    seen_utr: set[str] = set()
    rows = []

    for rec in records:
        description = _build_description(rec)
        raw_amount = rec.get("amount")
        if not description or raw_amount in (None, ""):
            continue
        try:
            amount = abs(float(str(raw_amount).replace(",", "").strip()))
        except (TypeError, ValueError):
            logger.warning(f"Skipping record with unparseable amount: {raw_amount!r}")
            continue

        # Dedupe on UTR / reference — a screenshot re-read yields the same UTR.
        utr = str(rec.get("utr", "")).strip()
        if utr:
            if utr in seen_utr:
                logger.info(f"  Skipping duplicate UTR {utr}")
                continue
            seen_utr.add(utr)

        tx_type = str(rec.get("type", "debit")).strip().lower()
        signed = amount if tx_type == "credit" else -amount
        rows.append({
            "date": str(rec.get("date", "")).strip() or today,
            "description": description,
            "amount": signed,
        })

    return pd.DataFrame(rows, columns=["date", "description", "amount"])



# ── Vision call (isolated so it can be mocked in tests) ───────────────────────

def _call_vision(image_bytes: bytes, mime_type: str) -> str:
    """Send the image + prompt to Gemini and return the raw response text."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    import os
    if not os.getenv("GEMINI_API_KEY"):
        raise ScreenshotExtractionError(
            "No GEMINI_API_KEY found. Screenshot extraction needs a Gemini key "
            "in your .env file (free tier works). CSV/PDF ingestion and offline "
            "Q&A still work without one."
        )

    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise ScreenshotExtractionError(
            "The google-genai package isn't installed. Run: pip install google-genai"
        ) from e

    client = genai.Client()
    last_error: Exception | None = None

    # Gemini's free tier occasionally returns transient 429/503 "high demand"
    # spikes. Retry a few times with backoff so a momentary blip doesn't fail a
    # live demo; give up (and surface a clear error) if it persists.
    for attempt in range(_VISION_RETRIES):
        try:
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    _PROMPT,
                ],
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                ),
            )
            return response.text or ""
        except Exception as e:  # noqa: BLE001 — classify then retry or raise
            if not _is_transient(e) or attempt == _VISION_RETRIES - 1:
                raise ScreenshotExtractionError(f"Gemini vision call failed: {e}") from e
            last_error = e
            logger.info(f"  Transient Gemini error ({e}); retrying…")
            time.sleep(_VISION_BACKOFF_SECONDS * (attempt + 1))

    raise ScreenshotExtractionError(f"Gemini vision call failed: {last_error}")


def _is_transient(exc: Exception) -> bool:
    """True for rate-limit / overload errors worth retrying (429, 500, 503)."""
    msg = str(exc)
    return any(code in msg for code in ("429", "500", "503", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))


# ── Public API ────────────────────────────────────────────────────────────────

_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".heic": "image/heic",
}


def extract_transactions_from_image(
    image_bytes: bytes, mime_type: str = "image/png"
) -> pd.DataFrame:
    """
    Read a payment screenshot (raw bytes) and return a clean transactions
    DataFrame: date, description, amount, transaction_type.

    Raises ScreenshotExtractionError if the key is missing, the call fails, or
    the image yielded no usable transaction.
    """
    if not image_bytes:
        raise ScreenshotExtractionError("Empty image — nothing to read.")

    logger.info(f"Extracting transactions from screenshot ({len(image_bytes)} bytes, {mime_type})")
    raw_text = _call_vision(image_bytes, mime_type)
    records = parse_transaction_json(raw_text)
    logger.info(f"  Vision model returned {len(records)} record(s)")

    raw_df = transactions_from_records(records)
    if raw_df.empty:
        raise ScreenshotExtractionError(
            "Couldn't read a transaction from that screenshot. "
            "Make sure the amount and payee are clearly visible."
        )

    return clean_transactions(raw_df)


def extract_transactions_from_file(filepath: str | Path) -> pd.DataFrame:
    """Convenience wrapper: read an image file from disk and extract."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Image file not found: {filepath}")
    mime_type = _MIME_BY_SUFFIX.get(filepath.suffix.lower(), "image/png")
    return extract_transactions_from_image(filepath.read_bytes(), mime_type)
