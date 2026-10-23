"""Rule-based question -> pandas code translator.

Used when no LLM API key is configured, so the app (and demos) always work.
It produces code strings that run through the same safe executor as the LLM path.
"""
from __future__ import annotations

import re

import pandas as pd

from profiler import column_types


def _mentioned(question: str, cols: list[str]) -> list[str]:
    q = question.lower()
    hits = []
    for c in cols:
        for name in {str(c).lower(), str(c).lower().replace("_", " ")}:
            m = re.search(r"\b" + re.escape(name) + r"\b", q)
            if m:
                hits.append((m.start(), c))
                break
    return [c for _, c in sorted(hits)]


def generate_code(question: str, df: pd.DataFrame) -> tuple[str, str]:
    """Return (explanation, code)."""
    q = question.lower()
    types = column_types(df)
    num, cat, dt = types["numeric"], types["categorical"], types["datetime"]
    mentioned = _mentioned(question, list(df.columns))
    m_num = [c for c in mentioned if c in num]
    m_cat = [c for c in mentioned if c in cat]
    m_dt = [c for c in mentioned if c in dt]

    agg = "mean" if re.search(r"\b(average|avg|mean)\b", q) else \
          "median" if "median" in q else \
          "max" if re.search(r"\b(max|maximum|highest)\b", q) else \
          "min" if re.search(r"\b(min|minimum|lowest)\b", q) else "sum"
    top = re.search(r"\btop\s+(\d+)", q)
    n = int(top.group(1)) if top else 5

    if re.search(r"missing|null|nan|empty", q):
        return ("Counting missing values per column.",
                "result = df.isna().sum().rename('missing').reset_index().rename(columns={'index': 'column'})\n"
                "result = result[result['missing'] > 0] if (result['missing'] > 0).any() else result\n"
                "fig = px.bar(result, x='column', y='missing', title='Missing values per column')")

    if re.search(r"correlat|relationship|heatmap", q):
        return ("Computing the correlation matrix of numeric columns.",
                "corr = df.select_dtypes('number').corr().round(2)\n"
                "result = corr\n"
                "fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', zmin=-1, zmax=1, "
                "title='Correlation heatmap')")

    if re.search(r"distribution|histogram|spread", q) and (m_num or num):
        c = (m_num or num)[0]
        return (f"Plotting the distribution of {c}.",
                f"result = df[{c!r}].describe().to_frame()\n"
                f"fig = px.histogram(df, x={c!r}, nbins=30, marginal='box', title='Distribution of {c}')")

    if re.search(r"trend|over time|monthly|by month|timeline", q) and (m_dt or dt) and (m_num or num):
        d, v = (m_dt or dt)[0], (m_num or num)[0]
        return (f"Aggregating {v} by month ({agg}) to show the trend.",
                f"result = df.groupby(df[{d!r}].dt.to_period('M').astype(str))[{v!r}].{agg}().reset_index()\n"
                f"fig = px.line(result, x={d!r}, y={v!r}, markers=True, title='{v} over time')")

    if (m_cat or cat) and (m_num or num) and re.search(
            r"\bby\b|per|each|top|best|worst|highest|lowest|total|average|compare|which", q):
        g = (m_cat or cat)[0]
        v = (m_num or num)[0]
        asc = bool(re.search(r"lowest|worst|bottom|least", q))
        return (f"Grouping by {g} and computing the {agg} of {v}.",
                f"result = df.groupby({g!r})[{v!r}].{agg}().sort_values(ascending={asc}).head({n}).reset_index()\n"
                f"fig = px.bar(result, x={g!r}, y={v!r}, title='{agg.title()} of {v} by {g}')")

    if re.search(r"count|how many|frequency|value counts", q) and (m_cat or cat):
        g = (m_cat or cat)[0]
        return (f"Counting rows for each {g}.",
                f"result = df[{g!r}].value_counts().head({n}).rename_axis({g!r}).reset_index(name='count')\n"
                f"fig = px.bar(result, x={g!r}, y='count', title='Count by {g}')")

    if len(m_num) >= 2 or (re.search(r"scatter|vs|versus", q) and len(num) >= 2):
        a, b = (m_num + num)[:2]
        return (f"Scatter plot of {a} vs {b}.",
                f"result = df[[{a!r}, {b!r}]].corr().round(3)\n"
                f"fig = px.scatter(df, x={a!r}, y={b!r}, trendline='ols', title='{a} vs {b}')")

    return ("No specific pattern matched, so here is a statistical summary of the dataset. "
            "Add an API key for full natural-language analysis.",
            "result = df.describe(include='all').T.reset_index().rename(columns={'index': 'column'})")
