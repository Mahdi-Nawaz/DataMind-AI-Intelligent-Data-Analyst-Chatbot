"""LLM-powered analysis: question -> pandas/Plotly code -> safe execution -> narrated answer.

Falls back to the offline rule-based engine when no API key is available or the
API call fails, so the app never dead-ends.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import pandas as pd

import offline_engine
from code_executor import ExecResult, run_code
from profiler import schema_text

DEFAULT_MODEL = os.getenv("DATAMIND_MODEL", "claude-sonnet-5-5")
MAX_RETRIES = 1

SYSTEM_PROMPT = """You are DataMind AI, an expert data analyst. The user has a pandas DataFrame named `df`.

Dataset description:
{schema}

Rules:
1. Answer the question by writing Python code inside ONE ```python block, preceded by a one-sentence plan.
2. Available names: df, pd, np, px (plotly.express), go (plotly.graph_objects). Do NOT import anything and do NOT read or write files.
3. Store tabular or scalar output in a variable named `result`.
4. When a chart helps, store a Plotly figure in a variable named `fig` with a clear title and axis labels.
5. Never modify the original `df` in a way the user did not ask for; use copies.
6. Use only column names that exist in the schema (they are case-sensitive).
7. If the question cannot be answered from the data, say so briefly and do not write code."""

NARRATE_PROMPT = """The user asked: {question}

The analysis produced this output:
{output}

Write a clear 2-4 sentence answer for a business reader. Lead with the key finding, quote specific numbers from the output, and mention one notable caveat or follow-up if relevant. Do not mention code."""


@dataclass
class Answer:
    text: str
    code: str | None = None
    exec_result: ExecResult | None = None
    mode: str = "llm"  # "llm" or "offline"
    error: str | None = None


def extract_code(text: str) -> str | None:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return m.group(1).strip() if m else None


def _text_of(response) -> str:
    return "".join(b.text for b in response.content if getattr(b, "type", "") == "text")


def _history_messages(history: list[dict]) -> list[dict]:
    msgs = []
    for h in history[-4:]:  # keep prompts small: last few turns only
        msgs.append({"role": "user", "content": h["question"]})
        msgs.append({"role": "assistant", "content": f"{h['text']}\n```python\n{h.get('code') or ''}\n```"})
    return msgs


def analyze(question: str, df: pd.DataFrame, history: list[dict], api_key: str | None,
            model: str = DEFAULT_MODEL) -> Answer:
    if not api_key:
        return _offline(question, df)
    try:
        return _llm(question, df, history, api_key, model)
    except Exception as e:  # noqa: BLE001 - network/auth/etc: degrade gracefully
        ans = _offline(question, df)
        ans.error = f"LLM unavailable ({type(e).__name__}); used offline mode instead."
        return ans


def _offline(question: str, df: pd.DataFrame) -> Answer:
    plan, code = offline_engine.generate_code(question, df)
    res = run_code(code, df)
    text = plan if res.ok else f"Offline analysis failed: {res.error}"
    return Answer(text=text, code=code, exec_result=res, mode="offline")


def _llm(question, df, history, api_key, model) -> Answer:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    system = SYSTEM_PROMPT.format(schema=schema_text(df))
    messages = _history_messages(history) + [{"role": "user", "content": question}]

    code, plan, res = None, "", None
    for attempt in range(MAX_RETRIES + 1):
        resp = client.messages.create(model=model, max_tokens=1500, system=system, messages=messages)
        reply = _text_of(resp)
        code = extract_code(reply)
        plan = reply.split("```")[0].strip()
        if code is None:  # model declined / answered in prose
            return Answer(text=reply.strip(), mode="llm")
        res = run_code(code, df)
        if res.ok:
            break
        # self-repair: show the model its error once
        messages += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": f"That code failed with: {res.error}\nFix it and return the corrected code."}]

    if not res.ok:
        return Answer(text="I couldn't produce a working analysis for that question.", code=code,
                      exec_result=res, mode="llm", error=res.error)

    narr = client.messages.create(
        model=model, max_tokens=400,
        messages=[{"role": "user", "content": NARRATE_PROMPT.format(question=question,
                                                                  output=res.summary_for_llm())}])
    return Answer(text=_text_of(narr).strip() or plan, code=code, exec_result=res, mode="llm")
