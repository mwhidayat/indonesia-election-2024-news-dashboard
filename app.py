# app_no_alias.py
import re
import json
import html
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

# Optional: nltk for sentence tokenization (used in KWIC fallback)
try:
    import nltk
    nltk.data.find("tokenizers/punkt")
except Exception:
    try:
        import nltk
        nltk.download("punkt", quiet=True)
    except Exception:
        pass

# -------------------------
# Page config & helpers
# -------------------------
st.set_page_config(page_title="Election News Dashboard",
                   page_icon="🗳️",
                   layout="wide",
                   initial_sidebar_state="expanded")

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", parse_dates=["Date"])
    required = {"Date", "Title", "Text", "Publication"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV missing columns: {required - set(df.columns)}")
    df["Title"] = df["Title"].fillna("").astype(str)
    df["Text"] = df["Text"].fillna("").astype(str)
    df["Publication"] = df["Publication"].fillna("Unknown").astype(str)
    df["Date_only"] = df["Date"].dt.normalize()
    return df

# -------------------------
# Load
# -------------------------
DATA_FILE = "indonesia-election-2024-dataset.csv"
df = load_data(DATA_FILE)

# -------------------------
# Sidebar filters
# -------------------------
st.sidebar.header("Filters")
min_date = df["Date_only"].min().date()
max_date = df["Date_only"].max().date()
date_range = st.sidebar.date_input("Date range", [min_date, max_date], min_value=min_date, max_value=max_date)

pubs = sorted(df["Publication"].unique())
selected_pubs = st.sidebar.multiselect("Publications", pubs, default=pubs)

# Candidate alias mapping (hardcoded)
default_aliases = {
    "Prabowo": ["prabowo"],
    "Ganjar": ["ganjar"],
    "Gibran": ["gibran"],
    "Anies": ["anies"],
    "Muhaimin": ["muhaimin", "imin", "gus imin", "cak imin"],
    "Amin": ["amin"],
    "Mahfud": ["mahfud"]
}
# "Edit aliases" UI removed per request. To change aliases, edit the default_aliases dict in the code.
candidate_aliases = default_aliases

all_candidates = list(candidate_aliases.keys())
selected_candidates = st.sidebar.multiselect("Candidates to display", all_candidates, default=all_candidates)

st.sidebar.markdown("---")
st.sidebar.caption("Results update live as you change filters. Use Search tab for detailed queries.")

# -------------------------
# Apply filters
# -------------------------
start_date, end_date = (date_range if len(date_range) == 2 else (min_date, max_date))
start_dt = pd.to_datetime(start_date)
end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

df_filtered = df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt) & (df["Publication"].isin(selected_pubs))].copy()

# -------------------------
# Utility functions
# -------------------------
def build_alias_pattern(aliases):
    esc = [re.escape(a) for a in aliases]
    return r"(?:" + r"|".join(esc) + r")"

def highlight_html(text: str, query: str, use_regex=False):
    """Return HTML-escaped text with <mark> around matches."""
    if not text:
        return ""
    escaped = html.escape(text)
    if use_regex:
        try:
            patt = re.compile(query, flags=re.IGNORECASE)
        except re.error:
            return escaped
        def repl(m):
            return f"<mark>{html.escape(m.group(0))}</mark>"
        return patt.sub(repl, html.escape(text))
    else:
        patt = re.compile(re.escape(query), flags=re.IGNORECASE)
        return patt.sub(lambda m: f"<mark>{html.escape(m.group(0))}</mark>", escaped)

def kwic_rows(text: str, query: str, window=7):
    """Return list of (left, keyword, right) occurrences for KWIC (word-window), case-insensitive."""
    if not text:
        return []
    words = re.split(r"\s+", text.strip())
    lowers = [w.lower() for w in words]
    q_tokens = re.split(r"\s+", query.strip().lower())
    q_len = len(q_tokens)
    rows = []
    for i in range(len(words)):
        segment = " ".join(lowers[i:i+q_len])
        if segment == " ".join(q_tokens):
            left_idx = max(0, i - window)
            right_idx = min(len(words), i + q_len + window)
            left = " ".join(words[left_idx:i])
            kw = " ".join(words[i:i+q_len])
            right = " ".join(words[i+q_len:right_idx])
            rows.append((left, kw, right))
    return rows

# -------------------------
# Tabs: Overview | Trends | Search | Key Word in Context | Data
# -------------------------
tabs = st.tabs(["Overview", "Trends", "Search", "Key Word in Context", "Data"])

####################
# OVERVIEW TAB
####################
with tabs[0]:
    st.header("Overview")
    # Use three compact metrics (removed 'Unique titles')
    c1, c2, c3 = st.columns(3)
    c1.metric("Total articles", f"{len(df_filtered):,}")
    c2.metric("Publications", f"{df_filtered['Publication'].nunique()}")
    # Always show the selected sidebar date range (clear and stable) instead of inferring from filtered data
    date_range_str = f"{pd.to_datetime(start_date).date()} → {pd.to_datetime(end_date).date()}"
    c3.metric("Selected date range", date_range_str)

    st.markdown("**Article distribution by publication**")
    article_counts = df_filtered["Publication"].value_counts().reset_index()
    article_counts.columns = ["Publication", "Count"]
    fig = px.bar(article_counts, x="Count", y="Publication", orientation="h", color="Publication",
                 color_discrete_sequence=px.colors.qualitative.Set2, title="Articles by Publication")
    fig.update_layout(showlegend=False, height=360)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Daily total (stacked by publication)**")
    timeseries = (df_filtered.groupby([pd.Grouper(key="Date", freq="D"), "Publication"])\
                  .size().reset_index(name="count"))
    if timeseries.empty:
        st.info("No data in the selected range / publications.")
    else:
        pivot = timeseries.pivot(index="Date", columns="Publication", values="count").fillna(0)
        pivot = pivot.resample("D").sum()
        y_cols = list(pivot.columns)
        fig_area = px.area(pivot.reset_index(), x="Date", y=y_cols, color_discrete_sequence=px.colors.qualitative.Set2)
        fig_area.update_layout(height=420)
        st.plotly_chart(fig_area, use_container_width=True)

####################
# TRENDS TAB
####################
with tabs[1]:
    st.header("Trends")
    st.markdown("Mentions of selected candidates in article titles over time (normalized by aliases).")

    for cand in selected_candidates:
        aliases = candidate_aliases.get(cand, [cand])
        patt = build_alias_pattern(aliases)
        df_filtered[f"cand__{cand}"] = df_filtered["Title"].str.contains(patt, flags=re.IGNORECASE, na=False, regex=True)

    cand_cols = [f"cand__{c}" for c in selected_candidates] if selected_candidates else []
    if cand_cols:
        cand_series = (df_filtered.groupby(pd.Grouper(key="Date", freq="D"))[cand_cols]
                       .sum().reset_index().melt(id_vars="Date", var_name="candidate", value_name="count"))
        cand_series["candidate"] = cand_series["candidate"].str.replace("^cand__", "", regex=True)
    else:
        cand_series = pd.DataFrame(columns=["Date", "candidate", "count"])

    if cand_series.empty or cand_series["count"].sum() == 0:
        st.info("No mentions of selected candidates in titles for current filters.")
    else:
        fig_cand = px.line(cand_series, x="Date", y="count", color="candidate",
                           color_discrete_sequence=px.colors.qualitative.Set2)
        fig_cand.update_traces(line_shape="spline")
        fig_cand.update_layout(height=420)
        st.plotly_chart(fig_cand, use_container_width=True)

        st.markdown("**Total mentions (bar)**")
        totals = cand_series.groupby("candidate")["count"].sum().reset_index().sort_values("count", ascending=True)
        fig_bar = px.bar(totals, x="count", y="candidate", orientation="h",
                         color="candidate", color_discrete_sequence=px.colors.qualitative.Set2)
        fig_bar.update_layout(showlegend=False, height=340)
        st.plotly_chart(fig_bar, use_container_width=True)

####################
# SEARCH TAB
####################
with tabs[2]:
    st.header("Search")
    st.markdown("Full-text search on Title + Text. Results show highlighted snippets and the option to download matches.")

    col_q1, col_q2, col_q3 = st.columns([4, 1, 1])
    query = col_q1.text_input("Search query (substring or regex)", placeholder="e.g. Prabowo OR \"Gus Imin\"")
    use_regex = col_q2.checkbox("Use regex", value=False)
    scope = col_q3.selectbox("Search in", ["Title + Text", "Title", "Text"], index=0)

    if query:
        patt = None
        if use_regex:
            try:
                patt = re.compile(query, flags=re.IGNORECASE)
            except re.error as e:
                st.error(f"Invalid regex: {e}")
                patt = None

        def matches_row(row):
            hay = ""
            if scope == "Title + Text":
                hay = f"{row['Title']} {row['Text']}"
            elif scope == "Title":
                hay = row["Title"]
            else:
                hay = row["Text"]
            if not isinstance(hay, str) or hay.strip() == "":
                return False
            if patt:
                return bool(patt.search(hay))
            return query.lower() in hay.lower()

        matched = df_filtered[df_filtered.apply(matches_row, axis=1)].copy().reset_index(drop=True)
        st.write(f"Found **{len(matched):,}** results")

        if not matched.empty:
            displayed = []
            for _, r in matched.iterrows():
                raw = (r["Title"] + ". " + r["Text"]).strip()
                if not raw:
                    continue
                if patt:
                    m = patt.search(raw)
                    if m:
                        start = max(0, m.start() - 120)
                        end = min(len(raw), m.end() + 120)
                        snippet_raw = raw[start:end]
                    else:
                        snippet_raw = raw[:300]
                    preview_html = highlight_html(snippet_raw, query, use_regex=True)
                else:
                    idx = raw.lower().find(query.lower())
                    if idx == -1:
                        snippet_raw = raw[:300]
                    else:
                        start = max(0, idx - 120)
                        end = min(len(raw), idx + 120)
                        snippet_raw = raw[start:end]
                    preview_html = highlight_html(snippet_raw, query, use_regex=False)

                preview_html = preview_html if len(html.unescape(preview_html)) <= 1000 else preview_html[:1000] + "..."
                displayed.append({
                    "Date": r["Date"].date(),
                    "Publication": r["Publication"],
                    "Title": html.escape(r["Title"]),
                    "Preview": preview_html,
                    "FullText": html.escape(r["Text"])
                })

            for item in displayed:
                title_for_expander = f"{item['Date']} — {item['Publication']} — {html.unescape(item['Title'])[:120]}"
                with st.expander(title_for_expander):
                    st.markdown(f"**Preview:**<br><div style='line-height:1.4'>{item['Preview']}</div>", unsafe_allow_html=True)
                    st.markdown("---")
                    st.write("Full text (raw):")
                    st.write(html.unescape(item["FullText"])[:3000] + ("..." if len(item["FullText"]) > 3000 else ""))

            st.download_button("Download matches as CSV", data=matched.to_csv(index=False), file_name="search_matches.csv", mime="text/csv")

####################
# KWIC TAB
####################
with tabs[3]:
    # Concordance / Key Word in Context function (supports single word or multi-word phrase)
    def display_concordance(data: pd.DataFrame, col: str, keyword: str, window_size: int = 7) -> pd.DataFrame:
        concordance_lines = []

        if not isinstance(keyword, str) or keyword.strip() == "":
            return pd.DataFrame(concordance_lines)

        # Prepare normalized keyword tokens (lowercased, stripped punctuation)
        kw_tokens = [re.sub(r"\W+", "", t).lower() for t in keyword.strip().split() if t.strip()]
        if not kw_tokens:
            return pd.DataFrame(concordance_lines)
        kw_len = len(kw_tokens)

        for index, row in data.iterrows():
            text = row.get(col, "")
            if not isinstance(text, str) or text.strip() == "":
                continue

            words = text.split()
            # normalized words for matching but keep original words for context
            norm_words = [re.sub(r"\W+", "", w).lower() for w in words]

            # slide over tokens to find phrase matches
            for i in range(0, max(1, len(words) - kw_len + 1)):
                try:
                    if all(norm_words[i + j] == kw_tokens[j] for j in range(kw_len)):
                        start = max(0, i - window_size)
                        end = min(len(words), i + kw_len + window_size)
                        left_context = ' '.join(words[start:i])
                        keyword_in_context = ' '.join(words[i:i + kw_len])
                        right_context = ' '.join(words[i + kw_len:end])

                        concordance_lines.append({
                            "Left context": left_context,
                            "Key Word": keyword_in_context,
                            "Right context": right_context,
                            "Publication": row.get("Publication", ""),
                            "Title": row.get("Title", "")

                        })
                except IndexError:
                    # defensive: skip if indexing goes out of range
                    continue
        return pd.DataFrame(concordance_lines)

    st.header("Key Word in Context")
    st.text("Explore occurrences of a keyword or phrase within the filtered data along with contextual snippets.")
    # Restrict selectable columns to Title or Text
    selected_column = st.selectbox("Choose column to search (Title or Text)", ["Text", "Title"])
    keyword = st.text_input("Enter a keyword or phrase (single word or multi-word phrase)")

    # Context window: presets + optional custom numeric input (more precise than a slider)
    preset_options = [3, 5, 7, 10, 15, 20, "Custom"]
    preset_index = 2  # default to 7
    choice = st.selectbox("Context window (words each side)", preset_options, index=preset_index)

    if choice == "Custom":
        window_size = int(st.number_input("Custom window size (words each side)", min_value=1, max_value=200, value=7, step=1))
    else:
        window_size = int(choice)

    st.caption("Presets for speed — choose Custom to type any number precisely.")

    if keyword:
        concordance_df = display_concordance(df_filtered, selected_column, keyword, window_size)
        st.write(f"### Key Word in Context for \"{keyword}\" in {selected_column}")
        if concordance_df.empty:
            st.info("No occurrences found in the filtered data.")
        else:
            st.dataframe(concordance_df, use_container_width=True)

####################
# DATA TAB
####################
with tabs[4]:
    st.header("Data Explorer")
    st.markdown("Preview and download the filtered dataset.")
    with st.expander("Preview (first 200 rows)"):
        st.dataframe(df_filtered[["Date", "Publication", "Title", "Text"]].head(200), use_container_width=True)
    st.download_button("Download filtered data (CSV)", data=df_filtered.to_csv(index=False), file_name="filtered_data.csv", mime="text/csv")

# Footer note
st.markdown("---")