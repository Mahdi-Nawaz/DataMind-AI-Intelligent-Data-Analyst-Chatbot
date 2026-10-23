# 🧠 DataMind AI — Intelligent Data Analyst Chatbot

An AI-powered data analysis app. Upload a CSV or Excel file, ask questions in plain English, and get answers as tables, interactive Plotly charts and written insights — no SQL or pandas knowledge required.

**Stack:** Python · Pandas · Streamlit · Plotly · Large Language Models (Anthropic Claude API)

## Features

- **Chat with your data** – natural-language questions become pandas + Plotly code, which is executed safely and explained in plain English. Every answer shows the generated code for transparency.
- **Self-repairing LLM loop** – if generated code fails, the error is fed back to the model for one automatic fix.
- **Offline mode** – no API key? A rule-based analyst still answers common questions (totals, averages, top-N, trends, correlations, missing values), so the app always works.
- **Data quality report** – missing %, unique counts, constant columns, duplicates.
- **Auto insights** – strongest correlation, skewed columns, dominant categories, written automatically.
- **Visual exploration + Chart Builder** – histograms, heatmaps, group-by charts, monthly trends, and a custom chart builder.
- **CSV / multi-sheet Excel support**, automatic date parsing, downloadable results.

## Architecture

```
Upload ─▶ data_loader ─▶ DataFrame
                          │
Question ─▶ llm_engine ───┤  schema + sample rows (never the full data) sent to the LLM
            │  (fallback: offline_engine)
            ▼
        generated pandas/Plotly code
            ▼
        code_executor  (AST validation + restricted builtins)
            ▼
   table + figure ─▶ LLM narrates the result ─▶ Streamlit UI
```

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI (chat, overview, quality, explore, chart builder) |
| `llm_engine.py` | Prompting, code extraction, retry loop, narration, fallback |
| `offline_engine.py` | Rule-based question → code translator |
| `code_executor.py` | Safe execution of generated code |
| `profiler.py` | Schema text, quality report, auto insights |
| `data_loader.py` | CSV/Excel loading and cleaning |
| `make_sample_data.py` | Sample sales dataset generator |
| `test_datamind.py` | Unit tests (pytest) |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# optional: enable LLM mode
export ANTHROPIC_API_KEY=your_key_here             # or paste it in the sidebar

streamlit run app.py
```

Click **Use sample sales data** in the sidebar, then try:

- "Show total sales by region"
- "What are the top 3 categories by profit?"
- "Show the monthly trend of sales"
- "Is there a correlation between discount and profit?"
- With an API key, anything: *"Which customer segment has the highest average order value, and is it growing?"*

Run tests: `pytest -q`

## Design decisions (good interview talking points)

1. **Privacy-aware prompting** – only column names, dtypes, a few example values and 5 sample rows go to the LLM, not the dataset.
2. **Code-generation over direct answering** – the LLM writes code that runs on the real data, so numbers are computed, not hallucinated, and the code is auditable.
3. **Safe execution** – generated code is parsed with `ast`; imports, dunder access, file I/O and `eval/exec/open` are rejected; only `df, pd, np, px, go` are in scope; the original DataFrame is never mutated. This is a pragmatic guard, not a hardened sandbox — production would use an isolated container/process with CPU/memory limits.
4. **Graceful degradation** – API failure or no key falls back to the offline engine.
5. **Tested** – 22 tests cover the executor's security rules, the offline engine, and the profiler.

## Roadmap

- Multi-file joins, SQL/database connectors
- Execution timeouts and process isolation
- Export a full analysis report as PDF
- Forecasting (Prophet / statsmodels) and anomaly detection
- Streaming responses and chat memory across sessions

## License

MIT
