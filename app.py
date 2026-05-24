"""
Indonesia Election 2024 News Dashboard
-------------------------------------
A Streamlit dashboard for exploring Indonesian 2024 election news coverage.

Expected CSV columns:
Date, Title, Text, URL, TextID, Publication

Default data file:
indonesia-election-2024-dataset.csv
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Indonesia Election 2024 News Dashboard",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

DATA_FILE = "indonesia-election-2024-dataset.csv"
REQUIRED_COLUMNS = {"Date", "Title", "Text", "Publication"}
OPTIONAL_COLUMNS = ["URL", "TextID"]

CANDIDATE_ALIASES: Dict[str, List[str]] = {
    "Anies–Muhaimin / AMIN": [
        "anies",
        "anies baswedan",
        "muhaimin",
        "cak imin",
        "gus imin",
        "amin",
    ],
    "Prabowo–Gibran": [
        "prabowo",
        "prabowo subianto",
        "gibran",
        "gibran rakabuming",
        "gibran rakabuming raka",
    ],
    "Ganjar–Mahfud": [
        "ganjar",
        "ganjar pranowo",
        "mahfud",
        "mahfud md",
    ],
}

INDIVIDUAL_ALIASES: Dict[str, List[str]] = {
    "Anies": ["anies", "anies baswedan"],
    "Muhaimin": ["muhaimin", "cak imin", "gus imin"],
    "Prabowo": ["prabowo", "prabowo subianto"],
    "Gibran": ["gibran", "gibran rakabuming", "gibran rakabuming raka"],
    "Ganjar": ["ganjar", "ganjar pranowo"],
    "Mahfud": ["mahfud", "mahfud md"],
}

ISSUE_TAXONOMY: Dict[str, List[str]] = {
    "Campaign & Mobilization": [
        "kampanye",
        "deklarasi",
        "relawan",
        "dukungan",
        "konsolidasi",
        "safari politik",
        "pemenangan",
    ],
    "Coalition & Party Politics": [
        "koalisi",
        "partai",
        "nasdem",
        "pdip",
        "gerindra",
        "golkar",
        "pkb",
        "pks",
        "pan",
        "demokrat",
        "psi",
    ],
    "Election Integrity & Pressure": [
        "tekanan",
        "intervensi",
        "intimidasi",
        "kecurangan",
        "netralitas",
        "bawaslu",
        "mahkamah konstitusi",
        "mk",
        "hukum",
        "kekuasaan",
    ],
    "Debate, Polling & Electability": [
        "debat",
        "survei",
        "elektabilitas",
        "polling",
        "hasil survei",
        "quick count",
        "real count",
    ],
    "Policy & Economy": [
        "ekonomi",
        "lapangan kerja",
        "pajak",
        "hilirisasi",
        "bansos",
        "subsidi",
        "pendidikan",
        "kesehatan",
        "program",
    ],
    "Governance & Institutions": [
        "presiden",
        "wakil presiden",
        "pemerintah",
        "kpu",
        "dpr",
        "menteri",
        "gubernur",
        "kabinet",
    ],
}

STOPWORDS_ID = {
    "yang",
    "dan",
    "di",
    "ke",
    "dari",
    "untuk",
    "dengan",
    "pada",
    "ini",
    "itu",
    "ada",
    "akan",
    "atau",
    "juga",
    "dalam",
    "sebagai",
    "karena",
    "oleh",
    "saat",
    "para",
    "kata",
    "jadi",
    "tak",
    "tidak",
    "bukan",
    "lebih",
    "telah",
    "usai",
    "soal",
    "terkait",
    "hingga",
    "agar",
    "bagi",
    "bisa",
    "hari",
    "tahun",
    "calon",
    "capres",
    "cawapres",
    "pilpres",
    "pemilu",
    "2024",
}

# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------

st.markdown(
    """
    <style>
    :root {
        --card-border: rgba(120, 120, 120, .20);
        --muted-text: rgba(120, 120, 120, .95);
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1280px;
    }
    .hero {
        border: 1px solid var(--card-border);
        border-radius: 24px;
        padding: 28px 30px;
        margin-bottom: 18px;
        background: linear-gradient(135deg, rgba(30, 64, 175, .12), rgba(220, 38, 38, .08));
    }
    .hero h1 {
        font-size: 2.15rem;
        line-height: 1.15;
        margin: 0 0 .45rem 0;
        letter-spacing: -0.03em;
    }
    .hero p {
        margin: 0;
        color: var(--muted-text);
        font-size: 1rem;
        max-width: 920px;
    }
    .metric-card {
        border: 1px solid var(--card-border);
        border-radius: 18px;
        padding: 18px 18px 14px 18px;
        min-height: 108px;
        background: rgba(255, 255, 255, .03);
    }
    .metric-label {
        color: var(--muted-text);
        font-size: .82rem;
        text-transform: uppercase;
        letter-spacing: .08em;
        margin-bottom: .5rem;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 750;
        line-height: 1.1;
    }
    .metric-note {
        color: var(--muted-text);
        font-size: .82rem;
        margin-top: .35rem;
    }
    .section-note {
        color: var(--muted-text);
        font-size: .94rem;
        margin-top: -.35rem;
        margin-bottom: .8rem;
    }
    mark {
        background: #fde68a;
        color: #111827;
        padding: 0 .18rem;
        border-radius: .25rem;
    }
    .small-caption {
        color: var(--muted-text);
        font-size: .85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Data loading and cleaning
# -----------------------------------------------------------------------------


def resolve_data_file() -> Path:
    """Return the local CSV file used by the dashboard.

    The app intentionally does not expose CSV upload in the UI. For deployment,
    place the dataset next to app.py. If DATA_FILE is not found and there is
    exactly one CSV in the app directory, that CSV is used as a fallback.
    """
    primary = Path(DATA_FILE)
    if primary.exists():
        return primary

    csv_files = sorted(Path(".").glob("*.csv"))
    if len(csv_files) == 1:
        return csv_files[0]

    if not csv_files:
        st.error(f"Dataset not found: `{DATA_FILE}`. Place the CSV next to `app.py`.")
    else:
        found = ", ".join(f"`{f.name}`" for f in csv_files[:8])
        st.error(
            f"Dataset not found: `{DATA_FILE}` and multiple CSV files are present ({found}). "
            f"Rename the intended dataset to `{DATA_FILE}`."
        )
    st.stop()


@st.cache_data(show_spinner=False)
def load_data_from_path(path: str) -> pd.DataFrame:
    return prepare_data(pd.read_csv(path, encoding="utf-8", on_bad_lines="skip"))


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {', '.join(sorted(missing))}")

    df = df.copy()
    for col in ["Title", "Text", "Publication", *OPTIONAL_COLUMNS]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.strip()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()

    df["Date_only"] = df["Date"].dt.normalize()
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    df["Title"] = df["Title"].replace({"nan": ""})
    df["Text"] = df["Text"].replace({"nan": ""})
    df["Publication"] = df["Publication"].replace({"": "Unknown", "nan": "Unknown"})
    df["Combined"] = (df["Title"] + " " + df["Text"]).str.strip()
    df["Word_Count"] = df["Text"].str.split().str.len().fillna(0).astype(int)

    return df.sort_values("Date").reset_index(drop=True)


# -----------------------------------------------------------------------------
# Text utilities
# -----------------------------------------------------------------------------


def alias_pattern(aliases: Iterable[str]) -> str:
    """Build a boundary-aware regex pattern for Indonesian news text."""
    cleaned = [a.strip() for a in aliases if isinstance(a, str) and a.strip()]
    cleaned = sorted(set(cleaned), key=len, reverse=True)
    if not cleaned:
        return r"a^"  # never matches
    body = "|".join(re.escape(alias) for alias in cleaned)
    return rf"(?<![\w])(?:{body})(?![\w])"


def contains_any(text: str, aliases: Iterable[str]) -> bool:
    if not isinstance(text, str) or not text:
        return False
    return bool(re.search(alias_pattern(aliases), text, flags=re.IGNORECASE))


def add_mentions(df: pd.DataFrame, alias_map: Dict[str, List[str]], prefix: str) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    cols: List[str] = []
    for label, aliases in alias_map.items():
        col = f"{prefix}{label}"
        out[col] = out["Combined"].str.contains(alias_pattern(aliases), case=False, regex=True, na=False)
        cols.append(col)
    return out, cols


def classify_issues(text: str) -> List[str]:
    labels = []
    for issue, keywords in ISSUE_TAXONOMY.items():
        if contains_any(text, keywords):
            labels.append(issue)
    return labels or ["Other / Unclassified"]


def highlight_text(text: str, query: str, use_regex: bool = False) -> str:
    safe_text = html.escape(str(text or ""))
    if not query:
        return safe_text

    try:
        pattern = re.compile(query if use_regex else re.escape(query), flags=re.IGNORECASE)
    except re.error:
        return safe_text

    raw_text = str(text or "")
    pieces = []
    last = 0
    for match in pattern.finditer(raw_text):
        pieces.append(html.escape(raw_text[last : match.start()]))
        pieces.append(f"<mark>{html.escape(match.group(0))}</mark>")
        last = match.end()
    pieces.append(html.escape(raw_text[last:]))
    return "".join(pieces)


def make_snippet(text: str, query: str, use_regex: bool = False, width: int = 180) -> str:
    raw = str(text or "")
    if not raw:
        return ""
    if not query:
        return raw[: width * 2]

    try:
        pattern = re.compile(query if use_regex else re.escape(query), flags=re.IGNORECASE)
    except re.error:
        return raw[: width * 2]

    match = pattern.search(raw)
    if not match:
        return raw[: width * 2]
    start = max(0, match.start() - width)
    end = min(len(raw), match.end() + width)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(raw) else ""
    return prefix + raw[start:end] + suffix


def kwic(data: pd.DataFrame, column: str, keyword: str, window: int = 7) -> pd.DataFrame:
    keyword = (keyword or "").strip()
    if not keyword:
        return pd.DataFrame()

    kw_tokens = [re.sub(r"\W+", "", token).lower() for token in keyword.split() if token.strip()]
    if not kw_tokens:
        return pd.DataFrame()

    rows = []
    kw_len = len(kw_tokens)

    for _, row in data.iterrows():
        text = str(row.get(column, "") or "")
        words = text.split()
        normalized = [re.sub(r"\W+", "", word).lower() for word in words]
        for idx in range(0, max(0, len(words) - kw_len + 1)):
            if normalized[idx : idx + kw_len] == kw_tokens:
                left = " ".join(words[max(0, idx - window) : idx])
                key = " ".join(words[idx : idx + kw_len])
                right = " ".join(words[idx + kw_len : idx + kw_len + window])
                rows.append(
                    {
                        "Date": row.get("Date_only"),
                        "Publication": row.get("Publication", ""),
                        "Title": row.get("Title", ""),
                        "Left context": left,
                        "Key word": key,
                        "Right context": right,
                    }
                )

    return pd.DataFrame(rows)


def top_terms(series: pd.Series, top_n: int = 25) -> pd.DataFrame:
    text = " ".join(series.dropna().astype(str).tolist()).lower()
    tokens = re.findall(r"[a-zA-ZÀ-ÿ0-9]+", text)
    tokens = [t for t in tokens if len(t) > 2 and t not in STOPWORDS_ID]
    if not tokens:
        return pd.DataFrame(columns=["Term", "Count"])
    counts = pd.Series(tokens).value_counts().head(top_n).reset_index()
    counts.columns = ["Term", "Count"]
    return counts.sort_values("Count", ascending=True)


def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{html.escape(label)}</div>
            <div class="metric-value">{html.escape(value)}</div>
            <div class="metric-note">{html.escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_headline_card(row: pd.Series) -> None:
    date_value = row.get("Date_only")
    date_str = pd.to_datetime(date_value).strftime("%Y-%m-%d") if pd.notna(date_value) else ""
    publication = html.escape(str(row.get("Publication", "Unknown")))
    title = html.escape(str(row.get("Title", "Untitled")))
    url = str(row.get("URL", "") or "").strip()
    text = html.escape(str(row.get("Text", ""))[:360])
    link = f"<a href='{html.escape(url)}' target='_blank'>Open source</a>" if url.startswith("http") else ""

    st.markdown(
        f"""
        <div style="border:1px solid var(--card-border); border-radius:16px; padding:14px 16px; margin-bottom:10px;">
            <div class="small-caption">{date_str} · {publication}</div>
            <div style="font-weight:700; margin:.25rem 0 .35rem 0;">{title}</div>
            <div class="small-caption">{text}{'…' if len(str(row.get('Text', ''))) > 360 else ''}</div>
            <div style="margin-top:.45rem;">{link}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Sidebar and data load
# -----------------------------------------------------------------------------

with st.sidebar:
    st.title("🗳️ Election News")
    st.caption("Filters apply to all tabs.")

try:
    data_path = resolve_data_file()
    df = load_data_from_path(str(data_path))
    data_source_label = data_path.name
except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()

if df.empty:
    st.warning("The dataset is empty after cleaning invalid dates.")
    st.stop()

min_date = df["Date_only"].min().date()
max_date = df["Date_only"].max().date()

with st.sidebar:
    date_range = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    publications = sorted(df["Publication"].dropna().unique().tolist())
    selected_publications = st.multiselect(
        "Publications",
        publications,
        default=publications,
    )

    selected_pairs = st.multiselect(
        "Candidate pairs",
        list(CANDIDATE_ALIASES.keys()),
        default=list(CANDIDATE_ALIASES.keys()),
    )

    selected_issues = st.multiselect(
        "Issue categories",
        list(ISSUE_TAXONOMY.keys()) + ["Other / Unclassified"],
        default=list(ISSUE_TAXONOMY.keys()) + ["Other / Unclassified"],
    )

    st.markdown("---")
    st.caption(f"Data source: `{data_source_label}`")
    st.caption("Mention counts are keyword/alias based, not stance or sentiment classification.")

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

start_dt = pd.to_datetime(start_date)
end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

filtered = df[
    (df["Date"] >= start_dt)
    & (df["Date"] <= end_dt)
    & (df["Publication"].isin(selected_publications))
].copy()

filtered, pair_cols = add_mentions(filtered, CANDIDATE_ALIASES, "pair__")
filtered, individual_cols = add_mentions(filtered, INDIVIDUAL_ALIASES, "person__")
filtered["Issue_List"] = filtered["Combined"].apply(classify_issues)
filtered["Primary_Issue"] = filtered["Issue_List"].str[0]

if selected_issues:
    filtered = filtered[filtered["Issue_List"].apply(lambda xs: any(x in selected_issues for x in xs))].copy()

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>Indonesia Election 2024 News Dashboard</h1>
        <p>
        A compact media-insight dashboard for exploring publication volume, candidate visibility,
        issue framing, search results, and keyword-in-context patterns in Indonesian election news.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------

overview_tab, candidate_tab, issue_tab, search_tab, kwic_tab, data_tab = st.tabs(
    ["Overview", "Candidate Landscape", "Issues & Framing", "Search", "KWIC", "Data"]
)

# -----------------------------------------------------------------------------
# Overview
# -----------------------------------------------------------------------------

with overview_tab:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Articles", f"{len(filtered):,}", "After active filters")
    with c2:
        metric_card("Publications", f"{filtered['Publication'].nunique():,}", "Unique sources")
    with c3:
        metric_card(
            "Date span",
            f"{pd.to_datetime(start_date).strftime('%d %b %Y')} → {pd.to_datetime(end_date).strftime('%d %b %Y')}",
            "Selected range",
        )
    with c4:
        avg_words = int(round(filtered["Word_Count"].mean(), 0)) if not filtered.empty else 0
        metric_card("Avg. article length", f"{avg_words:,}", "Words in Text column")

    st.subheader("Publication volume")
    st.markdown("<div class='section-note'>Compare source contribution and daily publication rhythm.</div>", unsafe_allow_html=True)

    left, right = st.columns([1, 1.25])
    with left:
        publication_counts = (
            filtered["Publication"].value_counts().reset_index(name="Articles").rename(columns={"index": "Publication"})
        )
        if publication_counts.empty:
            st.info("No data for the selected filters.")
        else:
            publication_counts = publication_counts.sort_values("Articles", ascending=True)
            fig = px.bar(
                publication_counts,
                x="Articles",
                y="Publication",
                orientation="h",
                text="Articles",
                title="Articles by publication",
            )
            fig.update_layout(height=380, showlegend=False, margin=dict(l=10, r=10, t=55, b=10))
            fig.update_traces(textposition="outside", cliponaxis=False)
            st.plotly_chart(fig, use_container_width=True)

    with right:
        daily = filtered.groupby([pd.Grouper(key="Date", freq="D"), "Publication"]).size().reset_index(name="Articles")
        if daily.empty:
            st.info("No daily trend available.")
        else:
            fig = px.area(
                daily,
                x="Date",
                y="Articles",
                color="Publication",
                title="Daily article volume by publication",
            )
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent / latest headlines in filtered data")
    for _, row in filtered.sort_values("Date", ascending=False).head(5).iterrows():
        render_headline_card(row)

# -----------------------------------------------------------------------------
# Candidate landscape
# -----------------------------------------------------------------------------

with candidate_tab:
    st.subheader("Candidate visibility")
    st.markdown(
        "<div class='section-note'>Mentions are counted from Title + Text using alias dictionaries. This measures visibility, not support or sentiment.</div>",
        unsafe_allow_html=True,
    )

    active_pair_cols = [f"pair__{label}" for label in selected_pairs if f"pair__{label}" in filtered.columns]
    if not active_pair_cols or filtered.empty:
        st.info("No candidate-pair data for the selected filters.")
    else:
        pair_daily = (
            filtered.groupby(pd.Grouper(key="Date", freq="D"))[active_pair_cols]
            .sum()
            .reset_index()
            .melt(id_vars="Date", var_name="Candidate Pair", value_name="Mentions")
        )
        pair_daily["Candidate Pair"] = pair_daily["Candidate Pair"].str.replace("pair__", "", regex=False)

        left, right = st.columns([1.25, 1])
        with left:
            fig = px.line(pair_daily, x="Date", y="Mentions", color="Candidate Pair", markers=True)
            fig.update_traces(line_shape="spline")
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=25, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with right:
            pair_totals = (
                pair_daily.groupby("Candidate Pair", as_index=False)["Mentions"]
                .sum()
                .sort_values("Mentions", ascending=True)
            )
            fig = px.bar(pair_totals, x="Mentions", y="Candidate Pair", orientation="h", text="Mentions")
            fig.update_layout(height=420, showlegend=False, margin=dict(l=10, r=10, t=25, b=10))
            fig.update_traces(textposition="outside", cliponaxis=False)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Individual figure mentions")
    if filtered.empty:
        st.info("No data for individual candidate mentions.")
    else:
        individual_totals = []
        for col in individual_cols:
            individual_totals.append(
                {
                    "Figure": col.replace("person__", ""),
                    "Articles mentioning figure": int(filtered[col].sum()),
                }
            )
        individual_df = pd.DataFrame(individual_totals).sort_values("Articles mentioning figure", ascending=True)
        fig = px.bar(
            individual_df,
            x="Articles mentioning figure",
            y="Figure",
            orientation="h",
            text="Articles mentioning figure",
        )
        fig.update_layout(height=420, showlegend=False, margin=dict(l=10, r=10, t=25, b=10))
        fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Candidate-pair co-mentions")
    if not active_pair_cols or filtered.empty:
        st.info("No co-mention matrix available.")
    else:
        labels = [col.replace("pair__", "") for col in active_pair_cols]
        matrix = pd.DataFrame(index=labels, columns=labels, dtype=int)
        for i, col_i in enumerate(active_pair_cols):
            for j, col_j in enumerate(active_pair_cols):
                matrix.iloc[i, j] = int((filtered[col_i] & filtered[col_j]).sum())
        fig = px.imshow(matrix, text_auto=True, aspect="auto", title="Article-level co-mentions")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Issues and framing
# -----------------------------------------------------------------------------

with issue_tab:
    st.subheader("Issue framing based on keyword taxonomy")
    st.markdown(
        "<div class='section-note'>Each article can match multiple issue categories. The primary issue is the first matched category in the taxonomy.</div>",
        unsafe_allow_html=True,
    )

    if filtered.empty:
        st.info("No issue data for the selected filters.")
    else:
        issue_rows = []
        for _, row in filtered.iterrows():
            for issue in row["Issue_List"]:
                issue_rows.append(
                    {
                        "Date": row["Date"],
                        "Publication": row["Publication"],
                        "Issue": issue,
                    }
                )
        issue_df = pd.DataFrame(issue_rows)

        left, right = st.columns([1, 1.25])
        with left:
            issue_counts = issue_df["Issue"].value_counts().reset_index(name="Articles")
            issue_counts = issue_counts.rename(columns={"index": "Issue"}).sort_values("Articles", ascending=True)
            fig = px.bar(issue_counts, x="Articles", y="Issue", orientation="h", text="Articles")
            fig.update_layout(height=430, showlegend=False, margin=dict(l=10, r=10, t=25, b=10))
            fig.update_traces(textposition="outside", cliponaxis=False)
            st.plotly_chart(fig, use_container_width=True)

        with right:
            issue_pub = issue_df.groupby(["Publication", "Issue"]).size().reset_index(name="Articles")
            fig = px.bar(
                issue_pub,
                x="Publication",
                y="Articles",
                color="Issue",
                title="Issue mix by publication",
            )
            fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10), xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Top headline terms")
        terms = top_terms(filtered["Title"], top_n=30)
        if terms.empty:
            st.info("No terms available.")
        else:
            fig = px.bar(terms, x="Count", y="Term", orientation="h", text="Count")
            fig.update_layout(height=560, showlegend=False, margin=dict(l=10, r=10, t=25, b=10))
            fig.update_traces(textposition="outside", cliponaxis=False)
            st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Search
# -----------------------------------------------------------------------------

with search_tab:
    st.subheader("Full-text search")
    st.markdown(
        "<div class='section-note'>Search the filtered dataset by title, body text, or both. Regex is optional.</div>",
        unsafe_allow_html=True,
    )

    q_col, regex_col, scope_col = st.columns([3, 1, 1])
    query = q_col.text_input("Search query", placeholder='e.g. "tekanan", Prabowo, Mahkamah Konstitusi')
    use_regex = regex_col.checkbox("Regex", value=False)
    scope = scope_col.selectbox("Search in", ["Title + Text", "Title", "Text"], index=0)

    if query:
        try:
            search_pattern = re.compile(query if use_regex else re.escape(query), flags=re.IGNORECASE)
        except re.error as exc:
            st.error(f"Invalid regex: {exc}")
            search_pattern = None

        if search_pattern is not None:
            def row_matches(row: pd.Series) -> bool:
                if scope == "Title":
                    haystack = row["Title"]
                elif scope == "Text":
                    haystack = row["Text"]
                else:
                    haystack = f"{row['Title']} {row['Text']}"
                return bool(search_pattern.search(str(haystack or "")))

            matches = filtered[filtered.apply(row_matches, axis=1)].copy()
            st.write(f"Found **{len(matches):,}** matching articles.")

            if not matches.empty:
                for _, row in matches.sort_values("Date", ascending=False).head(100).iterrows():
                    raw = f"{row['Title']}\n\n{row['Text']}"
                    snippet = make_snippet(raw, query, use_regex=use_regex)
                    title = f"{row['Date_only'].date()} — {row['Publication']} — {row['Title'][:120]}"
                    with st.expander(title):
                        st.markdown(highlight_text(snippet, query, use_regex=use_regex), unsafe_allow_html=True)
                        if str(row.get("URL", "")).startswith("http"):
                            st.link_button("Open source", row["URL"])
                        st.write(row["Text"][:4000] + ("…" if len(row["Text"]) > 4000 else ""))

                st.download_button(
                    "Download search results as CSV",
                    data=matches.drop(columns=[c for c in matches.columns if c.startswith(("pair__", "person__"))], errors="ignore").to_csv(index=False),
                    file_name="election_news_search_results.csv",
                    mime="text/csv",
                )
    else:
        st.info("Enter a query to search within the current filtered data.")

# -----------------------------------------------------------------------------
# KWIC
# -----------------------------------------------------------------------------

with kwic_tab:
    st.subheader("Keyword in Context")
    st.markdown(
        "<div class='section-note'>Use this to inspect how a word or phrase is used in the surrounding sentence context.</div>",
        unsafe_allow_html=True,
    )

    col_a, col_b, col_c = st.columns([2, 1, 1])
    kw_query = col_a.text_input("Keyword or phrase", placeholder="e.g. tekanan kekuasaan")
    kw_column = col_b.selectbox("Column", ["Text", "Title"], index=0)
    kw_window = col_c.number_input("Words each side", min_value=3, max_value=60, value=10, step=1)

    if kw_query:
        concordance = kwic(filtered, kw_column, kw_query, int(kw_window))
        st.write(f"Found **{len(concordance):,}** occurrences.")
        if concordance.empty:
            st.info("No occurrences found in the filtered data.")
        else:
            st.dataframe(concordance, use_container_width=True, height=520)
            st.download_button(
                "Download KWIC results as CSV",
                data=concordance.to_csv(index=False),
                file_name="election_news_kwic.csv",
                mime="text/csv",
            )
    else:
        st.info("Enter a keyword or phrase to generate concordance lines.")

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

with data_tab:
    st.subheader("Data explorer")
    st.markdown(
        "<div class='section-note'>Preview, inspect, and export the filtered dataset.</div>",
        unsafe_allow_html=True,
    )

    export = filtered.copy()
    export["Issue_List"] = export["Issue_List"].apply(lambda xs: "; ".join(xs) if isinstance(xs, list) else xs)
    drop_internal = [c for c in export.columns if c.startswith(("pair__", "person__"))]
    preview_cols = [c for c in ["Date", "Publication", "Title", "Text", "URL", "TextID", "Primary_Issue", "Issue_List"] if c in export.columns]

    st.dataframe(export[preview_cols].head(500), use_container_width=True, height=540)
    st.download_button(
        "Download filtered data as CSV",
        data=export.drop(columns=drop_internal, errors="ignore").to_csv(index=False),
        file_name="election_news_filtered.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption(
    "Method note: this dashboard uses rule-based keyword and alias matching for exploratory media insight. "
    "It should not be interpreted as sentiment, endorsement, or causal media-effect analysis."
)
