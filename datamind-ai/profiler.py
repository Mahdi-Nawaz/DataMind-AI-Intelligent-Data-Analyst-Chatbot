"""Dataset profiling: schema text for the LLM, data-quality report, and auto insights."""
from __future__ import annotations

import numpy as np
import pandas as pd


def schema_text(df: pd.DataFrame, sample_rows: int = 5) -> str:
    """Compact description of the dataframe that is sent to the LLM (never the full data)."""
    lines = [f"Rows: {len(df)}, Columns: {df.shape[1]}", "Columns (name | dtype | missing | examples):"]
    for col in df.columns:
        examples = df[col].dropna().unique()[:3]
        lines.append(f"- {col} | {df[col].dtype} | {int(df[col].isna().sum())} | {list(map(str, examples))}")
    lines.append("\nFirst rows:")
    lines.append(df.head(sample_rows).to_string(max_cols=20, max_colwidth=25))
    return "\n".join(lines)


def column_types(df: pd.DataFrame) -> dict[str, list[str]]:
    return {
        "numeric": df.select_dtypes("number").columns.tolist(),
        "categorical": df.select_dtypes(include=["object", "category", "bool"]).columns.tolist(),
        "datetime": df.select_dtypes("datetime").columns.tolist(),
    }


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column with missing %, unique count and a quality flag."""
    rows = []
    for col in df.columns:
        miss = df[col].isna().mean()
        nunique = df[col].nunique(dropna=True)
        flag = "OK"
        if miss > 0.3:
            flag = "High missing"
        elif nunique <= 1:
            flag = "Constant"
        elif miss > 0:
            flag = "Some missing"
        rows.append({"column": col, "dtype": str(df[col].dtype), "missing_%": round(miss * 100, 2),
                     "unique": nunique, "flag": flag})
    return pd.DataFrame(rows)


def auto_insights(df: pd.DataFrame) -> list[str]:
    """Plain-English findings generated with simple statistics (no LLM required)."""
    out = [f"The dataset has {len(df):,} rows and {df.shape[1]} columns."]

    dups = int(df.duplicated().sum())
    if dups:
        out.append(f"{dups:,} duplicate rows were found ({dups / len(df):.1%} of the data).")

    miss = df.isna().mean()
    if miss.max() > 0:
        out.append(f"'{miss.idxmax()}' has the most missing values ({miss.max():.1%}).")
    else:
        out.append("There are no missing values.")

    num = df.select_dtypes("number")
    if num.shape[1] >= 2:
        arr = np.nan_to_num(num.corr().abs().to_numpy(copy=True))
        np.fill_diagonal(arr, 0)
        i, j = np.unravel_index(arr.argmax(), arr.shape)
        if arr[i, j] > 0.3:
            a, b = num.columns[i], num.columns[j]
            r = num[a].corr(num[b])
            direction = "positive" if r > 0 else "negative"
            out.append(f"Strongest relationship: '{a}' and '{b}' have a {direction} correlation (r = {r:.2f}).")

    for col in num.columns[:6]:
        s = num[col].dropna()
        if len(s) > 10 and abs(s.skew()) > 1:
            out.append(f"'{col}' is skewed (skew = {s.skew():.1f}); the median ({s.median():,.2f}) "
                       f"differs noticeably from the mean ({s.mean():,.2f}).")
            break

    for col in df.select_dtypes(include=["object", "category"]).columns[:3]:
        vc = df[col].value_counts(normalize=True)
        if len(vc):
            out.append(f"Most common '{col}': {vc.index[0]} ({vc.iloc[0]:.1%} of rows).")
    return out
