import pandas as pd
import pytest

import offline_engine
import profiler
from code_executor import run_code, validate, UnsafeCodeError
from llm_engine import extract_code, analyze
from make_sample_data import make_sample


@pytest.fixture(scope="module")
def df():
    return make_sample(300)


def test_executor_runs_groupby_and_chart(df):
    res = run_code("result = df.groupby('region')['sales'].sum().reset_index()\n"
                   "fig = px.bar(result, x='region', y='sales')", df)
    assert res.ok and res.table is not None and res.fig is not None


def test_executor_does_not_mutate_original(df):
    before = df.shape
    run_code("df.drop(columns=['sales'], inplace=True)\nresult = df.shape", df)
    assert df.shape == before


@pytest.mark.parametrize("bad", [
    "import os",
    "from os import system",
    "open('x.txt')",
    "result = df.to_csv('a.csv')",
    "result = pd.read_csv('a.csv')",
    "result = ().__class__",
    "eval('1+1')",
])
def test_executor_blocks_unsafe_code(bad, df):
    with pytest.raises(UnsafeCodeError):
        validate(bad)
    assert not run_code(bad, df).ok


def test_runtime_error_is_reported(df):
    res = run_code("result = df['nope'].sum()", df)
    assert not res.ok and "KeyError" in res.error


@pytest.mark.parametrize("q", [
    "total sales by region", "average profit by category", "top 3 category by sales",
    "monthly trend of sales", "correlation between discount and profit",
    "missing values", "distribution of sales", "how many orders per region", "something odd",
])
def test_offline_engine_generates_runnable_code(q, df):
    _, code = offline_engine.generate_code(q, df)
    res = run_code(code, df)
    assert res.ok, f"{q!r} -> {res.error}\n{code}"


def test_analyze_without_key_uses_offline(df):
    ans = analyze("total sales by region", df, [], api_key=None)
    assert ans.mode == "offline" and ans.exec_result.ok


def test_extract_code():
    assert extract_code("plan\n```python\nresult = 1\n```") == "result = 1"
    assert extract_code("no code here") is None


def test_profiler(df):
    assert len(profiler.auto_insights(df)) >= 3
    assert set(profiler.quality_report(df)["column"]) == set(df.columns)
    assert "order_date" in profiler.schema_text(df)
