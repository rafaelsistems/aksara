import json
from pathlib import Path

import pytest

from tools.corpus_robustness_eval_large import run_evaluation


def test_large_evaluator_smoke_generates_json(tmp_path):
    output_json = tmp_path / "hasil.json"

    summary = run_evaluation(
        samples=10,
        progress_every=5,
        quiet_threshold=1,
        output_json=str(output_json),
    )

    assert output_json.exists()
    data = json.loads(output_json.read_text(encoding="utf-8"))

    assert data["run_size"] == 10
    assert data["valid_count"] <= 10
    assert 0.0 <= data["valid_rate"] <= 1.0
    assert "average_score" in data
    assert "timing" in data
    assert "category_aggregates" in data
    assert set(data["category_aggregates"].keys()) >= {"active", "passive", "anaphora", "nominal", "domain"}

    assert summary["run_size"] == 10
    assert summary["valid_count"] == data["valid_count"]


def test_large_evaluator_summary_structure():
    summary = run_evaluation(samples=5, progress_every=5, quiet_threshold=1)

    assert summary["run_size"] == 5
    assert 0.0 <= summary["valid_rate"] <= 1.0
    assert "timing" in summary
    assert "category_aggregates" in summary
    assert summary["category_aggregates"]["active"]["count"] >= 0
