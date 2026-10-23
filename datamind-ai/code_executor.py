"""Validate and execute LLM-generated pandas / Plotly code.

Safety model: the code is parsed with `ast` and rejected if it imports modules,
touches dunder attributes, or calls file/system-style functions. It then runs with a
whitelist of builtins and only `df`, `pd`, `np`, `px`, `go` in scope.

NOTE: this is a pragmatic guard for a demo/portfolio app, not a hardened sandbox.
For production, run generated code in a separate container or process with limits.
"""
from __future__ import annotations

import ast
import builtins
import contextlib
import io
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BLOCKED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "input", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "breakpoint", "exit", "quit", "help", "memoryview",
}
BLOCKED_ATTRS = {
    "read_csv", "read_excel", "read_pickle", "read_html", "read_json", "read_sql", "read_parquet",
    "to_csv", "to_excel", "to_pickle", "to_sql", "to_parquet", "to_json", "to_html",
    "write_html", "write_image", "write_json", "system", "popen",
}
_SAFE_BUILTIN_NAMES = [
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "isinstance",
    "len", "list", "map", "max", "min", "print", "range", "reversed", "round", "set", "slice",
    "sorted", "str", "sum", "tuple", "zip", "True", "False", "None",
]
SAFE_BUILTINS = {n: getattr(builtins, n) for n in _SAFE_BUILTIN_NAMES}

MAX_TABLE_ROWS = 1000


class UnsafeCodeError(Exception):
    pass


@dataclass
class ExecResult:
    ok: bool
    text: str = ""
    table: pd.DataFrame | None = None
    fig: go.Figure | None = None
    error: str | None = None
    stdout: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def summary_for_llm(self, limit: int = 3000) -> str:
        parts = []
        if self.table is not None:
            parts.append(self.table.head(20).to_string())
        if self.text:
            parts.append(self.text)
        if self.stdout:
            parts.append(self.stdout)
        if self.fig is not None:
            parts.append("(a chart was also produced)")
        return "\n".join(parts)[:limit]


def validate(code: str) -> ast.AST:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise UnsafeCodeError(f"Syntax error: {e}") from e
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise UnsafeCodeError("Imports are not allowed; pd, np, px and go are already available.")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise UnsafeCodeError(f"Access to '{node.attr}' is not allowed.")
            if node.attr in BLOCKED_ATTRS:
                raise UnsafeCodeError(f"Use of '{node.attr}' is not allowed.")
        if isinstance(node, ast.Name) and (node.id in BLOCKED_NAMES or node.id.startswith("__")):
            raise UnsafeCodeError(f"Use of '{node.id}' is not allowed.")
    return tree


def run_code(code: str, df: pd.DataFrame) -> ExecResult:
    """Execute code against a copy of df. Code should set `result` and/or `fig`."""
    try:
        tree = validate(code)
    except UnsafeCodeError as e:
        return ExecResult(ok=False, error=str(e))

    ns: dict[str, Any] = {"df": df.copy(), "pd": pd, "np": np, "px": px, "go": go,
                          "__builtins__": SAFE_BUILTINS}
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(tree, "<analysis>", "exec"), ns)  # noqa: S102 - validated above
    except Exception as e:  # noqa: BLE001 - surface any runtime error to the user / LLM
        return ExecResult(ok=False, error=f"{type(e).__name__}: {e}", stdout=buf.getvalue())

    result = ns.get("result")
    fig = ns.get("fig")
    table, text = None, ""
    if isinstance(result, pd.Series):
        result = result.to_frame()
    if isinstance(result, pd.DataFrame):
        table = result.head(MAX_TABLE_ROWS)
    elif result is not None:
        text = str(result)
    if not isinstance(fig, go.Figure):
        fig = None
    return ExecResult(ok=True, text=text, table=table, fig=fig, stdout=buf.getvalue())
