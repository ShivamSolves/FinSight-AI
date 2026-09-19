"""
app.py — FinSight AI, a beautiful Streamlit front-end for grounded finance RAG.

Run it:
    streamlit run app/app.py

Flow
────
  upload / sample CSV  →  load_csv → categorize_dataframe → build_vector_store
  Ask tab     → retrieve() (+ ask() when an LLM key is present, else a
                deterministic local answer) so the demo always works offline
  Dashboard   → Altair charts: spend by category, monthly trend, top merchants
  Transactions→ filterable, searchable table
"""

from __future__ import annotations

import os
import tempfile

import altair as alt
import pandas as pd
import streamlit as st

from finsight.categorization.categorizer import categorize_dataframe
from finsight.config import CURRENCY_SYMBOL as S
from finsight.config import RAW_DIR
from finsight.embeddings.vector_store import build_vector_store
from finsight.ingestion.csv_extractor import load_csv
from finsight.ingestion.pdf_extractor import load_pdf
from finsight.rag.local_answer import summarize_hits
from finsight.rag.qa_pipeline import ask, retrieve

# ── Palette ──────────────────────────────────────────────────────────────────
# Wondercraft-inspired: mint primary, teal-blue secondary, pastel chips on a
# pale periwinkle canvas with navy ink.
INDIGO, VIOLET, FUCHSIA = "#2f7494", "#6672d8", "#f3a9d3"
GREEN, ROSE, SLATE = "#009f7a", "#e0567a", "#64748f"
CATEGORY_COLORS = [
    "#00cf9e", "#2f7494", "#7fb5e6", "#bcc7f6", "#f3a9d3", "#ffd166",
    "#8fe9cb", "#5a8ea6", "#e0567a", "#6672d8", "#aed7f2", "#f3cbe7", "#a3b1c6",
]

st.set_page_config(
    page_title="FinSight AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ────────────────────────────────────────────────────────────────────
# The <style> tag must be the FIRST token of its own markdown call: markdown-it
# only treats <style> as a raw HTML block when it opens a line, otherwise the
# CSS leaks onto the page as visible text.
st.markdown(
    """<style>
    html, body, [class*="css"] { font-family: 'Figtree', -apple-system, 'Segoe UI', sans-serif; }
    #MainMenu, footer, header {visibility: hidden;}
    .stApp {background: transparent;}
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1240px;}
    h1,h2,h3,h4,h5,h6 {
        font-family:'Bricolage Grotesque', 'Figtree', sans-serif;
        color:#233048 !important; letter-spacing:-.02em;
    }

    /* ── pastel wondercraft background ──────────────────────────────────── */
    .fs-bg {position:fixed; inset:0; z-index:-1; overflow:hidden; background:#f5f6fc;}
    .fs-bg .wash {position:absolute; border-radius:50%; filter:blur(110px);}
    .fs-bg .w1 {width:560px; height:560px; left:-160px; top:-180px; background:#8fe9cb; opacity:.35;
                animation: fs-float 18s ease-in-out infinite;}
    .fs-bg .w2 {width:520px; height:520px; right:-150px; bottom:-170px; background:#aed7f2; opacity:.40;
                animation: fs-float 22s ease-in-out infinite reverse;}
    .fs-bg .chip {position:absolute; border-radius:26px; opacity:.45;
                  animation: fs-bob 14s ease-in-out infinite;}
    .fs-bg .c1 {width:120px; height:86px; left:7%;  top:16%; background:#aed7f2; transform:rotate(-8deg);}
    .fs-bg .c2 {width:96px;  height:70px; left:79%; top:10%; background:#8fe9cb; transform:rotate(7deg);
                animation-delay:-4s;}
    .fs-bg .c3 {width:110px; height:78px; left:66%; top:64%; background:#f3cbe7; transform:rotate(-6deg);
                animation-delay:-8s;}
    .fs-bg .c4 {width:84px;  height:62px; left:15%; top:72%; background:#bcc7f6; transform:rotate(9deg);
                animation-delay:-11s;}
    @keyframes fs-float {0%,100% {transform:translate(0,0);} 50% {transform:translate(40px,-30px);}}
    @keyframes fs-bob  {0%,100% {margin-top:0;} 50% {margin-top:-22px;}}

    /* ── hero ─────────────────────────────────────────────────────────────── */
    .hero {
        position:relative; overflow:hidden; border-radius:26px;
        padding:2.4rem 2.6rem; margin-bottom:1.5rem;
        background:#ffffff; border:1px solid #e9ecf7;
        box-shadow:0 24px 60px -30px rgba(35,48,72,.25);
    }
    .hero::after {
        content:""; position:absolute; top:0; left:0; right:0; height:4px;
        background:linear-gradient(90deg,#00cf9e,#7fb5e6,#f3a9d3);
    }
    .hero h1 {position:relative; font-size:2.7rem; font-weight:700; margin:0;}
    .hero h1 a, .hero a {display:none;}
    .hero p  {position:relative; color:#64748f; font-size:1.04rem; margin:.6rem 0 0; font-weight:400;}
    .hero .tag {
        position:relative; display:inline-block; color:#009f7a; background:#e4faf3;
        border:1px solid #b5ecd9; padding:.26rem .8rem; border-radius:999px;
        font-size:.72rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase;
        margin-bottom:.9rem;
    }

    /* ── KPI cards ────────────────────────────────────────────────────────── */
    .kpi {
        height:100%; background:#ffffff; border:1px solid #e9ecf7;
        border-radius:20px; padding:1.2rem 1.35rem;
        box-shadow:0 14px 34px -24px rgba(35,48,72,.28);
    }
    .kpi .label {color:#7a8699; font-size:.72rem; font-weight:700;
                 text-transform:uppercase; letter-spacing:.12em;}
    .kpi .value {font-family:'Bricolage Grotesque', 'Figtree', sans-serif;
                 font-size:1.72rem; font-weight:700; margin-top:.35rem; letter-spacing:-.02em;}
    .kpi .sub   {font-size:.78rem; color:#7a8699; margin-top:.2rem;}

    /* ── badges ───────────────────────────────────────────────────────────── */
    .scope-badge {
        display:inline-block; background:#e4faf3; color:#009f7a; border:1px solid #b5ecd9;
        border-radius:999px; padding:.24rem .7rem; font-size:.74rem; font-weight:700;
        margin-right:.4rem;
    }
    .conf-badge {
        display:inline-block; border-radius:999px; padding:.24rem .7rem; font-size:.74rem;
        font-weight:700; border:1px solid transparent; margin-right:.4rem;
    }
    .conf-high    {background:#e4faf3; color:#009f7a; border-color:#b5ecd9;}
    .conf-med     {background:#fdf3df; color:#a16207; border-color:#f3e0b0;}
    .conf-neutral {background:#eef0fa; color:#64748f; border-color:#dfe3f2;}

    .brand-side {font-family:'Bricolage Grotesque', 'Figtree', sans-serif;
        font-weight:700; font-size:1.4rem; letter-spacing:-.02em; color:#233048;}
    .side-sub {color:#7a8699; font-size:.84rem; margin:.15rem 0 1.2rem;}

    /* ── sidebar / tabs / controls ────────────────────────────────────────── */
    section[data-testid="stSidebar"] {background:#ffffff; border-right:1px solid #e9ecf7;}
    .stTabs [data-baseweb="tab"] {font-weight:600; color:#7a8699;}
    .stTabs [aria-selected="true"] {color:#233048 !important;}
    .stTabs [data-baseweb="tab-highlight"] {background:#00cf9e !important;}
    .stButton>button {
        background:#ffffff; border:1px solid #dfe3f2; color:#233048;
        border-radius:10px; font-weight:600; transition:all .18s ease;
    }
    .stButton>button:hover {background:#e4faf3; border-color:#00cf9e; color:#009f7a;}
    section[data-testid="stSidebar"] .stButton>button {
        background:#00cf9e; border-color:#00cf9e; color:#ffffff;
    }
    section[data-testid="stSidebar"] .stButton>button:hover {
        background:#00b78b; border-color:#00b78b; color:#ffffff;
    }
    .stTextInput input, .stTextInput input:focus {
        background:#ffffff; border:1px solid #dfe3f2; color:#233048;
    }
    .stTextInput input:focus {border-color:#00cf9e; box-shadow:0 0 0 3px rgba(0,207,158,.16);}
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color:#e9ecf7 !important; border-radius:18px; background:#ffffff;
        box-shadow:0 14px 34px -26px rgba(35,48,72,.25);
    }
    [data-testid="stArrowVegaLiteChart"] {
        background:#ffffff; border:1px solid #e9ecf7; border-radius:20px;
        padding:.9rem 0 .5rem; box-shadow:0 14px 34px -26px rgba(35,48,72,.22);
    }
    div[data-testid="stMetric"] {background:transparent;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700;800'
    '&family=Bricolage+Grotesque:wght@400;500;600;700&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="fs-bg" aria-hidden="true">'
    '<div class="wash w1"></div><div class="wash w2"></div>'
    '<div class="chip c1"></div><div class="chip c2"></div>'
    '<div class="chip c3"></div><div class="chip c4"></div></div>',
    unsafe_allow_html=True,
)


# ── Data loading (cached) ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_upload(name: str, data: bytes) -> pd.DataFrame:
    tmp = os.path.join(tempfile.gettempdir(), f"finsight_{name}")
    with open(tmp, "wb") as f:
        f.write(data)
    loader = load_pdf if name.lower().endswith(".pdf") else load_csv
    return categorize_dataframe(loader(tmp))


@st.cache_data(show_spinner=False)
def _load_sample(path_str: str) -> pd.DataFrame:
    return categorize_dataframe(load_csv(path_str))


def ensure_store(df: pd.DataFrame) -> None:
    """Build the vector store once per distinct dataset (embeddings are costly)."""
    sig = f"{len(df)}_{df['amount'].sum():.2f}_{df['date'].min()}_{df['date'].max()}"
    if st.session_state.get("store_sig") != sig:
        with st.spinner("Embedding transactions into the vector store…"):
            build_vector_store(df)
        st.session_state["store_sig"] = sig


def has_llm_key() -> bool:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    return bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))


def money(x: float) -> str:
    return f"{S}{x:,.0f}"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="brand-side">◈ FinSight AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-sub">Grounded answers over your bank statement</div>',
                unsafe_allow_html=True)

    st.markdown("**1 · Data source**")
    uploaded = st.file_uploader("Upload a statement (CSV or PDF)", type=["csv", "pdf"],
                                label_visibility="collapsed")
    sample_path = RAW_DIR / "bank_statement.csv"
    if st.button("✨ Load sample statement", use_container_width=True,
                 disabled=not sample_path.exists()):
        st.session_state["use_sample"] = True

    st.divider()
    st.markdown("**2 · Engine**")
    if has_llm_key():
        st.markdown('<span class="scope-badge">🟢 LLM connected</span> '
                    '<span style="font-size:.8rem;color:#7a8699">answers via GPT</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="scope-badge">⚡ Offline mode</span> '
                    '<span style="font-size:.8rem;color:#7a8699">deterministic local answers</span>',
                    unsafe_allow_html=True)

    st.divider()
    st.caption("Retrieval is exact — aggregate questions fetch the complete "
               "matching set, not a top-K sample.")

# ── Resolve the active dataframe ───────────────────────────────────────────────
df = None
if uploaded is not None:
    st.session_state["use_sample"] = False
    try:
        df = _load_upload(uploaded.name, uploaded.getvalue())
        source_label = uploaded.name
    except Exception as exc:
        source_label = None
        st.error(
            f"Couldn't read **{uploaded.name}** as a bank statement.\n\n"
            f"{exc}\n\n"
            "Expected columns: a date, a description/merchant, and an amount "
            "(or separate debit/credit columns). Remove the file in the sidebar "
            "or load the sample instead."
        )
elif st.session_state.get("use_sample") and sample_path.exists():
    df = _load_sample(str(sample_path))
    source_label = "Sample bank statement"
else:
    source_label = None

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
      <span class="tag">Personal Finance · RAG</span>
      <h1>Ask your money anything.</h1>
      <p>FinSight reads a messy bank statement, categorizes every transaction,
         and answers questions with grounded, auditable totals — no hallucinated numbers.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if df is None:
    st.markdown(
        '<div style="text-align:center;padding:3rem 1rem;color:#7a8699">'
        '<div style="font-size:3rem">📊</div>'
        '<h3 style="font-weight:700;margin:.4rem 0">Load a statement to begin</h3>'
        '<p style="max-width:440px;margin:0 auto">Use the sidebar to upload your own CSV, '
        'or click <b style="color:#009f7a">Load sample statement</b> to explore a demo dataset instantly.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# Store ready — safe to query.
ensure_store(df)

# ── Derived numbers ────────────────────────────────────────────────────────────
spend_df = df[df["amount"] < 0].copy()
spend_df["abs_amount"] = spend_df["amount"].abs()
income_df = df[df["amount"] > 0].copy()

total_spend = float(spend_df["abs_amount"].sum())
total_income = float(income_df["amount"].sum())
net = total_income - total_spend
n_txn = len(df)
months = sorted(df["date"].dt.to_period("M").unique())
date_span = f"{df['date'].min():%b %Y} – {df['date'].max():%b %Y}"

st.markdown(f'<div style="color:#7a8699;font-size:.86rem;margin:-.4rem 0 1rem;">'
            f'📄 <b style="color:#233048">{source_label}</b> · {n_txn} transactions · {date_span}'
            f'</div>', unsafe_allow_html=True)


def kpi(label, value, sub, color):
    return (f'<div class="kpi"><div class="label">{label}</div>'
            f'<div class="value" style="color:{color}">{value}</div>'
            f'<div class="sub">{sub}</div></div>')


c1, c2, c3, c4 = st.columns(4)
c1.markdown(kpi("Income", money(total_income), f"{len(income_df)} credits", GREEN), unsafe_allow_html=True)
c2.markdown(kpi("Spending", money(total_spend), f"{len(spend_df)} debits", ROSE), unsafe_allow_html=True)
c3.markdown(kpi("Net", money(net), "income − spending", INDIGO if net >= 0 else ROSE), unsafe_allow_html=True)
c4.markdown(kpi("Avg / month", money(total_spend / max(len(months), 1)),
                f"over {len(months)} month{'s' if len(months)!=1 else ''}", VIOLET), unsafe_allow_html=True)

st.markdown("<div style='height:1.1rem'></div>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
ask_tab, dash_tab, txn_tab = st.tabs(["💬  Ask FinSight", "📈  Dashboard", "🧾  Transactions"])

# ·· ASK ··
with ask_tab:
    st.markdown("##### Try a question")
    suggestions = [
        "What was my total grocery spend in February 2026?",
        "How much did I spend at Swiggy?",
        "Show all my subscriptions across all months",
        "What was my total income?",
        "How much did I spend on transport in January?",
    ]
    cols = st.columns(len(suggestions))
    for col, q in zip(cols, suggestions):
        if col.button(q, use_container_width=True, key=f"sug_{q}"):
            st.session_state["question"] = q

    question = st.text_input(
        "Your question",
        key="question",
        placeholder="e.g. How much did I spend on dining out in February?",
        label_visibility="collapsed",
    )

    if question:
        with st.spinner("Retrieving matching transactions…"):
            hits, parsed, complete = retrieve(question)

        if not hits:
            st.info("No transactions matched that question. Try a different merchant, "
                    "category, or month.")
        else:
            local = summarize_hits(hits, question, complete=complete)
            conf_class = "conf-high" if local.confidence == "High" else "conf-med"
            st.markdown(
                f'<div style="margin:.4rem 0">'
                f'<span class="scope-badge">🔎 {parsed.describe_scope()}</span>'
                f'<span class="conf-badge {conf_class}">{local.confidence} confidence</span>'
                f'<span class="conf-badge conf-neutral">'
                f'{local.count} transaction{"s" if local.count!=1 else ""}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            if has_llm_key():
                with st.spinner("Asking the model…"):
                    result = ask(question)
                if result.error:
                    st.warning(f"LLM unavailable ({result.error}) — showing the computed answer.")
                    answer_md = local.to_markdown(question)
                else:
                    answer_md = result.answer
            else:
                answer_md = local.to_markdown(question)

            with st.container(border=True):
                st.markdown(answer_md)

            with st.expander(f"🧾 Evidence — the {local.count} transactions used"):
                ev = pd.DataFrame([
                    {
                        "Date": h["metadata"]["date"],
                        "Description": h["metadata"]["description"],
                        "Category": h["metadata"]["category"],
                        "Amount": h["metadata"]["amount"],
                    } for h in local.evidence
                ])
                st.dataframe(ev, use_container_width=True, hide_index=True)

# ·· DASHBOARD ··
with dash_tab:
    left, right = st.columns([1, 1])

    with left:
        st.markdown("##### Spending by category")
        cat = spend_df.groupby("category")["abs_amount"].sum().reset_index()
        cat = cat.sort_values("abs_amount", ascending=False)
        donut = (
            alt.Chart(cat)
            .mark_arc(innerRadius=62, outerRadius=110, stroke="#ffffff", strokeWidth=2)
            .encode(
                theta=alt.Theta("abs_amount:Q"),
                color=alt.Color("category:N", legend=None,
                                scale=alt.Scale(range=CATEGORY_COLORS)),
                tooltip=["category:N", alt.Tooltip("abs_amount:Q", format=",.0f")],
            )
            .properties(height=320)
        )
        total_lbl = alt.Chart(pd.DataFrame({"t": [total_spend]})).mark_text(
            text=money(total_spend), fontSize=20, fontWeight=700, color="#233048", dy=-6
        ).properties(height=320)
        sub_lbl = alt.Chart(pd.DataFrame({"t": [total_spend]})).mark_text(
            text="total spend", fontSize=12, color="#7a8699", dy=16
        ).properties(height=320)
        st.altair_chart((donut + total_lbl + sub_lbl).configure(background="transparent"),
                        use_container_width=True)

        legend = "  ".join(
            f'<span style="color:{CATEGORY_COLORS[i % len(CATEGORY_COLORS)]}">●</span> {c}'
            for i, c in enumerate(cat["category"])
        )
        st.markdown(f'<div style="font-size:.78rem;line-height:1.9">{legend}</div>',
                    unsafe_allow_html=True)

    with right:
        st.markdown("##### Cash flow by month")
        tmp = df.copy()
        tmp["month"] = tmp["date"].dt.to_period("M").dt.to_timestamp()
        tmp["signed_spend"] = tmp["amount"].clip(upper=0).abs()
        monthly = tmp.groupby("month").apply(
            lambda g: pd.Series({
                "Spending": g["signed_spend"].sum(),
                "Income": g["amount"].clip(lower=0).sum(),
            }), include_groups=False
        ).reset_index()
        long = monthly.melt("month", var_name="Flow", value_name="Amount")
        trend = (
            alt.Chart(long)
            .mark_area(interpolate="monotone", line=True, opacity=0.85)
            .encode(
                x=alt.X("month:T", title=None),
                y=alt.Y("Amount:Q", title=None, axis=alt.Axis(format="~s")),
                color=alt.Color("Flow:N", scale=alt.Scale(
                    domain=["Spending", "Income"], range=[ROSE, GREEN]), legend=None),
                tooltip=["month:T", "Flow:N", alt.Tooltip("Amount:Q", format=",.0f")],
            )
            .properties(height=300)
            .configure(background="transparent")
        )
        st.altair_chart(trend, use_container_width=True)
        monthly["net"] = monthly["Income"] - monthly["Spending"]
        peak_spend_month = monthly.loc[monthly["Spending"].idxmax(), "month"]
        best_save_month = monthly.loc[monthly["net"].idxmax(), "month"]
        ml, mr = st.columns(2)
        ml.markdown(f'<div class="kpi"><div class="label">Peak spend month</div>'
                    f'<div class="value" style="font-size:1.2rem;color:{ROSE}">'
                    f'{peak_spend_month:%b %Y}</div></div>',
                    unsafe_allow_html=True)
        mr.markdown(f'<div class="kpi"><div class="label">Best saving month</div>'
                    f'<div class="value" style="font-size:1.2rem;color:{GREEN}">'
                    f'{best_save_month:%b %Y}</div></div>',
                    unsafe_allow_html=True)

    st.markdown("##### Top merchants")
    merch = spend_df.groupby("description")["abs_amount"].sum().reset_index()
    merch = merch.sort_values("abs_amount", ascending=False).head(8)
    bars = (
        alt.Chart(merch)
        .mark_bar(cornerRadius=6)
        .encode(
            x=alt.X("abs_amount:Q", title=None, axis=alt.Axis(format="~s")),
            y=alt.Y("description:N", sort="-x", title=None),
            color=alt.Color("description:N", legend=None,
                            scale=alt.Scale(range=[INDIGO, VIOLET, FUCHSIA])),
            tooltip=["description:N", alt.Tooltip("abs_amount:Q", format=",.0f")],
        )
        .properties(height=320)
        .configure(background="transparent")
    )
    st.altair_chart(bars, use_container_width=True)

# ·· TRANSACTIONS ··
with txn_tab:
    f1, f2 = st.columns([2, 1])
    with f1:
        search = st.text_input("🔍 Search description", placeholder="swiggy, rent, salary…",
                               label_visibility="collapsed")
    with f2:
        cats = st.multiselect("Category", sorted(df["category"].unique()),
                              label_visibility="collapsed", placeholder="All categories")

    view = df.copy()
    if search:
        view = view[view["description"].str.contains(search, case=False, na=False)]
    if cats:
        view = view[view["category"].isin(cats)]
    view = view.sort_values("date", ascending=False)

    st.caption(f"Showing {len(view)} of {n_txn} transactions")
    display = pd.DataFrame({
        "Date": view["date"].dt.strftime("%Y-%m-%d"),
        "Description": view["description"],
        "Category": view["category"],
        "Type": view["transaction_type"],
        "Amount": view["amount"].map(lambda x: f"{S}{x:,.2f}"),
    })
    st.dataframe(
        display, use_container_width=True, hide_index=True, height=520,
        column_config={
            "Category": st.column_config.TextColumn("Category", width="medium"),
            "Amount": st.column_config.TextColumn("Amount", width="small"),
        },
    )
