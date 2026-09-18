"""
vector_store.py — Embeds transaction chunks and stores/queries them in ChromaDB.

Architecture
────────────
Each transaction row becomes one "chunk" — a structured natural language
sentence that packs every dimension a query might reference:

    "{Category} transaction on {Weekday, Month DD YYYY}: {Description}
     — ${amount}. Type: {debit/credit}."

e.g.:
    "Groceries transaction on Monday, January 05 2026:
     Wholesome Market - Groceries — $87.43. Type: debit."

Why this format?
  A bare description like "Uber Technologies" has nothing for a query
  like "transport spending in February" to match against. By including
  the category, full date, amount, and type in the chunk text, the
  vector model has all the relevant dimensions to compute similarity
  against a natural language question.

Embedding model: sentence-transformers/all-MiniLM-L6-v2
  - Runs fully locally, no API key needed
  - ~80 MB download on first use (cached after that)
  - 384-dimensional embeddings, fast inference

Vector store: ChromaDB (persistent, file-based)
  - Stored at data/chroma_db/ (configured in config.py)
  - Collection name: "transactions"
  - Each document also stores metadata (date, amount, category,
    transaction_type) so we can do exact numeric lookups alongside
    semantic retrieval

Public API
──────────
  build_vector_store(df)      → indexes all transactions, returns collection
  query_vector_store(q, n=10) → returns top-n most relevant transactions
  get_or_create_collection()  → returns existing collection (cheap, no re-embed)
"""

import logging

import pandas as pd

from finsight.config import CURRENCY_SYMBOL as _CURRENCY_SYMBOL

logger = logging.getLogger(__name__)

# ── Lazy imports — these are heavy; only pay the cost when the module is used ─
_embedder = None
_chroma_client = None


def _get_embedder():
    """Load the sentence-transformer model once and reuse it."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        from finsight.config import EMBEDDING_MODEL
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("  Embedding model loaded ✓")
    return _embedder


def _get_chroma_client():
    """Create or reuse the ChromaDB persistent client."""
    global _chroma_client
    if _chroma_client is None:
        import chromadb

        from finsight.config import CHROMA_DIR
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        logger.info(f"ChromaDB client initialised → {CHROMA_DIR}")
    return _chroma_client


# ── Chunk builder ─────────────────────────────────────────────────────────────

def build_chunk_text(row: pd.Series) -> str:
    """
    Convert a single transaction row into the embedding-ready text chunk.

    This is the exact format agreed on:
        "{Category} transaction on {Weekday, Month DD YYYY}: {Description}
         — ${amount}. Type: {debit/credit}."

    Args:
        row: A single row from the categorized transactions DataFrame.
             Must have: date, description, amount, transaction_type, category.

    Returns:
        Formatted string ready to be embedded.
    """
    day_str = row["date"].strftime("%A, %B %d %Y")
    amount  = abs(row["amount"])
    return (
        f"{row['category']} transaction on {day_str}: "
        f"{row['description']} — {_CURRENCY_SYMBOL}{amount:.2f}. "
        f"Type: {row['transaction_type']}."
    )


# ── Build (index) ─────────────────────────────────────────────────────────────

def build_vector_store(df: pd.DataFrame, collection_name: str = "transactions"):
    """
    Embed all transactions and store them in ChromaDB.

    If a collection with this name already exists it is deleted and rebuilt
    from scratch. This keeps the store consistent with the current CSV —
    if you re-run ingestion with new data, just call this again.

    Args:
        df: Categorized transactions DataFrame (output of categorize_dataframe).
            Must have: date, description, amount, transaction_type, category.
        collection_name: ChromaDB collection name (default: "transactions").

    Returns:
        The ChromaDB collection object (can be passed to query functions).
    """
    client   = _get_chroma_client()
    embedder = _get_embedder()

    # Delete existing collection so a re-run starts clean
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)
        logger.info(f"Deleted existing collection '{collection_name}' for rebuild")

    collection = client.create_collection(
        name=collection_name,
        # cosine distance is better than L2 for semantic similarity on
        # normalised sentence-transformer embeddings
        metadata={"hnsw:space": "cosine"},
    )

    # ── Build chunk texts and metadata ───────────────────────────────────────
    chunks    = []
    metadatas = []
    ids       = []

    for idx, row in df.iterrows():
        chunk_text = build_chunk_text(row)
        chunks.append(chunk_text)

        # Store rich metadata alongside the vector so the Q&A layer can do
        # exact-match filtering (e.g. filter by month) without re-embedding
        metadatas.append({
            "date":             str(row["date"].date()),
            "month":            row["date"].strftime("%Y-%m"),       # "2026-01"
            "month_name":       row["date"].strftime("%B %Y"),       # "January 2026"
            "month_num":        int(row["date"].month),              # 1-12, year-agnostic filter
            "year":             int(row["date"].year),               # 2026
            "description":      row["description"],
            "amount":           float(row["amount"]),
            "abs_amount":       float(abs(row["amount"])),
            "transaction_type": row["transaction_type"],
            "category":         row["category"],
        })

        # ChromaDB requires string IDs; use row index padded for readability
        ids.append(f"txn_{idx:04d}")

    # ── Embed in one batch (much faster than row-by-row) ─────────────────────
    logger.info(f"Embedding {len(chunks)} transaction chunks...")
    embeddings = embedder.encode(
        chunks,
        batch_size=64,          # all-MiniLM handles 64 at a time easily
        show_progress_bar=False,
        convert_to_numpy=True,
    ).tolist()                  # ChromaDB expects plain Python lists
    logger.info("  Embeddings computed ✓")

    # ── Upsert into ChromaDB ──────────────────────────────────────────────────
    # Add in batches of 100 (ChromaDB recommendation for large inserts)
    batch_size = 100
    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=chunks[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )

    logger.info(f"  Stored {len(chunks)} chunks in collection '{collection_name}' ✓")
    return collection


# ── Query ─────────────────────────────────────────────────────────────────────

def query_vector_store(
    query_text: str,
    n_results: int = None,
    collection_name: str = "transactions",
    where: dict | None = None,
    merchant: str | None = None,
):
    """
    Retrieve the most relevant transactions for a natural language query.

    Two retrieval modes, chosen by the caller (usually qa_pipeline via the
    query parser):

      • Semantic top-K (default): pure vector similarity, good for open-ended
        "show me X" questions where completeness doesn't matter.
      • Filtered/complete: pass a ChromaDB `where` clause (e.g. category and/or
        month) plus a large `n_results` so EVERY matching transaction is
        returned. This is what aggregate questions ("total grocery spend in
        February") need — a top-K cutoff would silently drop rows and make the
        LLM's total wrong.

    Args:
        query_text:      The user's question in plain English (used for ranking).
        n_results:       Max rows to return. None → TOP_K_RESULTS from config.
                         Pass a large value (or get_collection_count()) for
                         complete retrieval.
        collection_name: ChromaDB collection to query (default: "transactions").
        where:           Optional ChromaDB metadata filter, e.g.
                         {"category": "Groceries"} or
                         {"$and": [{"category": "Groceries"}, {"month_num": 2}]}.
        merchant:        Optional merchant keyword. ChromaDB has no substring
                         operator, so this is applied as a post-filter on the
                         description text (case-insensitive).

    Returns:
        List of dicts, each with keys:
            id, document (chunk text), metadata (date/amount/category/etc.), distance
        Sorted by relevance (lowest cosine distance first).
    """
    from finsight.config import TOP_K_RESULTS

    if n_results is None:
        n_results = TOP_K_RESULTS

    client   = _get_chroma_client()
    embedder = _get_embedder()

    collection = client.get_collection(collection_name)

    # ChromaDB warns/errors if n_results exceeds the number of rows it can
    # return, so cap it at the collection size.
    total_docs = collection.count()
    if total_docs:
        n_results = min(n_results, total_docs)

    # Embed the query with the same model used for the chunks
    query_embedding = embedder.encode([query_text], convert_to_numpy=True).tolist()

    query_kwargs = {
        "query_embeddings": query_embedding,
        "n_results":        n_results,
        "include":          ["documents", "metadatas", "distances"],
    }
    if where:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)

    # Flatten ChromaDB's nested-list response into a cleaner list of dicts
    hits = []
    for doc, meta, dist, rid in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        results["ids"][0],
    ):
        hits.append({
            "id":       rid,
            "document": doc,
            "metadata": meta,
            "distance": round(dist, 4),  # cosine distance: 0 = identical, 2 = opposite
        })

    # Merchant post-filter (ChromaDB `where` can't do substring matching).
    if merchant:
        needle = merchant.lower()
        hits = [h for h in hits if needle in h["metadata"]["description"].lower()]

    return hits


def get_collection_count(collection_name: str = "transactions") -> int:
    """Number of documents in the collection (0 if it doesn't exist yet)."""
    client = _get_chroma_client()
    existing = [c.name for c in client.list_collections()]
    if collection_name not in existing:
        return 0
    return client.get_collection(collection_name).count()



def get_or_create_collection(collection_name: str = "transactions"):
    """
    Return an existing ChromaDB collection without re-embedding.
    Useful when the store has already been built and you just want to query.

    Raises ValueError if the collection doesn't exist yet (run build_vector_store first).
    """
    client     = _get_chroma_client()
    existing   = [c.name for c in client.list_collections()]
    if collection_name not in existing:
        raise ValueError(
            f"Collection '{collection_name}' not found. "
            "Run build_vector_store(df) first to index your transactions."
        )
    return client.get_collection(collection_name)
