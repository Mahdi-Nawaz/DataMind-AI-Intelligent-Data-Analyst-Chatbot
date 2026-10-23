"""Loading and light cleaning of uploaded CSV / Excel files."""
from __future__ import annotations

import os

import pandas as pd

SUPPORTED = (".csv", ".xlsx", ".xls")


def load_file(uploaded) -> dict[str, pd.DataFrame]:
    """Return {sheet_name: DataFrame} for an uploaded file-like object.

    CSV files return a single entry named after the file.
    Excel files return one entry per sheet.
    """
    name = uploaded.name
    lower = name.lower()
    if not lower.endswith(SUPPORTED):
        raise ValueError(f"Unsupported file type. Please upload one of: {', '.join(SUPPORTED)}")

    if lower.endswith(".csv"):
        try:
            df = pd.read_csv(uploaded)
        except UnicodeDecodeError:
            uploaded.seek(0)
            df = pd.read_csv(uploaded, encoding="latin-1")
        return {os.path.splitext(name)[0]: clean(df)}

    sheets = pd.read_excel(uploaded, sheet_name=None)
    return {sheet: clean(df) for sheet, df in sheets.items() if not df.empty}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Strip column names, drop fully empty rows/cols, and parse obvious date columns."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)

    for col in df.columns:
        if df[col].dtype == object and any(k in col.lower() for k in ("date", "time", "day", "month")):
            parsed = pd.to_datetime(df[col], errors="coerce")
            # only convert when most values parse successfully
            if parsed.notna().mean() > 0.8:
                df[col] = parsed
    return df
