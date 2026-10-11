import json
from pathlib import Path

from eval.harness import load_eval_items, run_eval, summarize_eval_items


def test_eval_harness_loads_question_dataset() -> None:
    items = load_eval_items(Path(__file__).resolve().parents[1] / "eval" / "questions.jsonl")

    assert len(items) >= 4
    assert {item.category for item in items} >= {"direct_lookup", "date_sensitive", "numeric", "abstain"}


def test_eval_harness_summary_and_run_eval_are_consistent(tmp_path) -> None:
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "question": "sample question",
                        "ref_date": "2025-01-01",
                        "gold_paths": ["CGST/s17"],
                        "gold_answer": "sample answer",
                        "category": "direct_lookup",
                        "answerable": True,
                        "source": "hand",
                    }
                ),
                json.dumps(
                    {
                        "question": "unanswerable question",
                        "ref_date": "2025-01-01",
                        "gold_paths": [],
                        "gold_answer": "do not guess",
                        "category": "abstain",
                        "answerable": False,
                        "source": "hand",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    items = load_eval_items(dataset)
    summary = summarize_eval_items(items)
    assert summary["num_items"] == 2
    assert summary["answerable"] == 1
    assert summary["unanswerable"] == 1

    results_file = tmp_path / "results.jsonl"
    record = run_eval(dataset, results_file, config_name="test-config")
    assert record["config"] == "test-config"
    assert len(results_file.read_text(encoding="utf-8").strip().splitlines()) == 1
