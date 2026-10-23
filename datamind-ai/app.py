"""DataMind AI - Intelligent Data Analyst Chatbot (Streamlit UI).

Run:  streamlit run app.py
"""
from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import streamlit as st

import data_loader
import llm_engine
import profiler
from make_sample_data import make_sample

st.set_page_config(page_title="DataMind AI", page_icon="🧠", layout="wide")

EXAMPLES = [
    "Show total sales by region",
    "What are the top 3 categories by profit?",
    "Show the monthly trend of sales",
    "Is there a correlation between discount and profit?",
    "How many missing values are in each column?",
]


# ----------------------------------------------------------------------------- state
def init_state():
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("dataset_key", None)


def render_message(msg: dict, idx: int):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("error"):
            st.caption(f"⚠️ {msg['error']}")
        res = msg.get("exec_result")
        if res is not None and res.ok:
            if res.fig is not None:
                st.plotly_chart(res.fig, width="stretch", key=f"fig_{idx}")
            if res.table is not None:
                st.dataframe(res.table, width="stretch")
                st.download_button("Download result (CSV)", res.table.to_csv(index=False).encode(),
                                   f"result_{idx}.csv", "text/csv", key=f"dl_{idx}")
        if msg.get("code"):
            with st.expander("View generated code"):
                st.code(msg["code"], language="python")


# ----------------------------------------------------------------------------- sidebar
def sidebar() -> tuple[pd.DataFrame | None, str | None, str]:
    st.sidebar.title("🧠 DataMind AI")
    st.sidebar.caption("Ask questions about your data in plain English.")

    uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    use_sample = st.sidebar.button("Use sample sales data")
    if use_sample:
        st.session_state["use_sample"] = True
    if uploaded is not None:
        st.session_state["use_sample"] = False

    df, key = None, None
    if uploaded is not None:
        try:
            sheets = data_loader.load_file(uploaded)
        except Exception as e:  # noqa: BLE001
            st.sidebar.error(f"Could not read file: {e}")
            sheets = {}
        if sheets:
            sheet = st.sidebar.selectbox("Sheet", list(sheets)) if len(sheets) > 1 else next(iter(sheets))
            df, key = sheets[sheet], f"{uploaded.name}:{sheet}"
    elif st.session_state.get("use_sample"):
        df, key = make_sample(), "sample"

    if key != st.session_state["dataset_key"]:  # new dataset -> fresh conversation
        st.session_state["dataset_key"] = key
        st.session_state["messages"] = []

    st.sidebar.divider()
    api_key = st.sidebar.text_input("Anthropic API key (optional)", type="password",
                                    value=os.getenv("ANTHROPIC_API_KEY", ""),
                                    help="Without a key the app uses a built-in offline analyst.")
    model = st.sidebar.text_input("Model", value=llm_engine.DEFAULT_MODEL)
    st.sidebar.success("LLM mode" if api_key else "Offline mode (rule-based)")
    if st.sidebar.button("Clear chat"):
        st.session_state["messages"] = []
    return df, (api_key or None), model


# ----------------------------------------------------------------------------- tabs
def tab_overview(df: pd.DataFrame):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    c4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
    st.subheader("Preview")
    st.dataframe(df.head(100), width="stretch")
    st.subheader("Statistical summary")
    st.dataframe(df.describe(include="all").T, width="stretch")


def tab_quality(df: pd.DataFrame):
    st.subheader("Data quality report")
    st.dataframe(profiler.quality_report(df), width="stretch")
    st.subheader("Auto insights")
    for line in profiler.auto_insights(df):
        st.markdown(f"- {line}")


def tab_explore(df: pd.DataFrame):
    types = profiler.column_types(df)
    num, cat, dt = types["numeric"], types["categorical"], types["datetime"]
    st.subheader("Quick visual exploration")
    left, right = st.columns(2)
    if num:
        col = left.selectbox("Histogram of", num, key="hist_col")
        left.plotly_chart(px.histogram(df, x=col, marginal="box", title=f"Distribution of {col}"),
                          width="stretch")
    if len(num) >= 2:
        corr = df[num].corr().round(2)
        right.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                     title="Correlation heatmap"), width="stretch")
    if cat and num:
        c, v = st.columns(2)
        g = c.selectbox("Group by", cat, key="grp")
        m = v.selectbox("Measure", num, key="meas")
        agg = st.radio("Aggregation", ["sum", "mean", "median", "count"], horizontal=True)
        out = df.groupby(g)[m].agg(agg).reset_index().sort_values(m, ascending=False)
        st.plotly_chart(px.bar(out, x=g, y=m, title=f"{agg.title()} of {m} by {g}"), width="stretch")
    if dt and num:
        d = st.selectbox("Date column", dt, key="dtcol")
        m = st.selectbox("Trend measure", num, key="trendm")
        out = df.groupby(df[d].dt.to_period("M").astype(str))[m].sum().reset_index()
        st.plotly_chart(px.line(out, x=d, y=m, markers=True, title=f"Monthly {m}"), width="stretch")


def tab_chat(df: pd.DataFrame, api_key: str | None, model: str):
    st.subheader("Chat with your data")
    if not st.session_state["messages"]:
        st.caption("Try one of these:")
        cols = st.columns(len(EXAMPLES))
        for i, ex in enumerate(EXAMPLES):
            if cols[i].button(ex, key=f"ex_{i}", width="stretch"):
                st.session_state["pending"] = ex
                st.rerun()

    for i, m in enumerate(st.session_state["messages"]):
        render_message(m, i)

    question = st.chat_input("Ask a question about your data…") or st.session_state.pop("pending", None)
    if question:
        st.session_state["messages"].append({"role": "user", "content": question})
        history = [
            {"question": a["question"], "text": a["content"], "code": a.get("code")}
            for a in st.session_state["messages"] if a["role"] == "assistant" and a.get("question")
        ]
        with st.spinner("Analysing…"):
            ans = llm_engine.analyze(question, df, history, api_key, model)
        st.session_state["messages"].append({
            "role": "assistant", "content": ans.text, "code": ans.code, "exec_result": ans.exec_result,
            "error": ans.error, "question": question,
        })
        st.rerun()


def tab_builder(df: pd.DataFrame):
    st.subheader("Chart builder")
    cols = df.columns.tolist()
    c1, c2, c3, c4 = st.columns(4)
    kind = c1.selectbox("Chart type", ["bar", "line", "scatter", "histogram", "box", "pie"])
    x = c2.selectbox("X", cols)
    y = c3.selectbox("Y", [None] + cols)
    color = c4.selectbox("Color", [None] + cols)
    try:
        if kind == "bar":
            fig = px.bar(df, x=x, y=y, color=color)
        elif kind == "line":
            fig = px.line(df.sort_values(x), x=x, y=y, color=color)
        elif kind == "scatter":
            fig = px.scatter(df, x=x, y=y, color=color)
        elif kind == "histogram":
            fig = px.histogram(df, x=x, color=color)
        elif kind == "box":
            fig = px.box(df, x=x, y=y, color=color)
        else:
            fig = px.pie(df, names=x, values=y)
        st.plotly_chart(fig, width="stretch")
    except Exception as e:  # noqa: BLE001
        st.warning(f"Could not draw that chart: {e}")


# ----------------------------------------------------------------------------- main
def main():
    init_state()
    df, api_key, model = sidebar()
    st.title("DataMind AI — Intelligent Data Analyst")

    if df is None:
        st.info("👈 Upload a CSV/Excel file, or click **Use sample sales data** to try the app.")
        st.markdown("**What you can do:** explore data quality, auto-generate insights and charts, "
                    "build custom charts, and ask questions in plain English.")
        return

    t1, t2, t3, t4, t5 = st.tabs(["💬 Chat", "📋 Overview", "🩺 Quality & Insights", "📊 Explore", "🛠 Chart Builder"])
    with t1:
        tab_chat(df, api_key, model)
    with t2:
        tab_overview(df)
    with t3:
        tab_quality(df)
    with t4:
        tab_explore(df)
    with t5:
        tab_builder(df)


main()
