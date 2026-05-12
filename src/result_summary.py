import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"


def load_result_files(results_dir: Path) -> list[Path]:
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")
    paths = sorted(path for path in results_dir.iterdir() if path.suffix == ".json")
    if not paths:
        raise FileNotFoundError(f"No json result files found in {results_dir}")
    return paths


def iter_result_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if "results" in payload:
        return payload["results"]

    # Compatibility with the original result shape.
    benchmark = payload.get("benchmark_dataset", [])
    inference = payload.get("inference_results", [])
    judge = payload.get("judge_results")
    logicality = payload.get("logical_nexus_results") or payload.get("rubric_results")
    if logicality is None:
        raise ValueError("Result file must contain either results or logical_nexus_results")

    rows = []
    for idx, item in enumerate(benchmark[: len(logicality)]):
        row = {
            "uid": item.get("uid", idx),
            "question_type": item.get("question_type"),
            "judge_result": judge[idx] if judge is not None and idx < len(judge) else None,
            "logicality_result": logicality[idx],
        }
        if idx < len(inference):
            row["answer_pred"] = inference[idx].get("answer_pred")
        rows.append(row)
    return rows


def new_bucket() -> dict[str, float]:
    return {
        "num_examples": 0,
        "num_accuracy_examples": 0,
        "accuracy_sum": 0.0,
        "F_sum": 0.0,
        "O_sum": 0.0,
        "P_sum": 0.0,
        "recall_sum": 0.0,
        "precision_sum": 0.0,
    }


def add_row(bucket: dict[str, float], row: dict[str, Any]) -> None:
    logicality = row.get("logicality_result")
    if not logicality:
        raise ValueError(f"Missing logicality_result for uid={row.get('uid')}")

    bucket["num_examples"] += 1
    bucket["F_sum"] += float(logicality.get("F_score", 0.0))
    bucket["O_sum"] += float(logicality.get("O_score", 0.0))
    bucket["P_sum"] += float(logicality.get("P_score", 0.0))
    bucket["recall_sum"] += float(logicality.get("recall", 0.0))
    bucket["precision_sum"] += float(logicality.get("precision", 0.0))

    judge_result = row.get("judge_result")
    if judge_result is not None:
        bucket["num_accuracy_examples"] += 1
        bucket["accuracy_sum"] += float(judge_result.get("correct", 0.0))


def finalize_bucket(bucket: dict[str, float]) -> dict[str, float | int | None]:
    n = int(bucket["num_examples"])
    acc_n = int(bucket["num_accuracy_examples"])
    if n == 0:
        return {
            "num_examples": 0,
            "num_accuracy_examples": 0,
            "accuracy": None,
            "F": None,
            "O": None,
            "P": None,
            "recall": None,
            "precision": None,
        }
    return {
        "num_examples": n,
        "num_accuracy_examples": acc_n,
        "accuracy": bucket["accuracy_sum"] / acc_n if acc_n else None,
        "F": bucket["F_sum"] / n,
        "O": bucket["O_sum"] / n,
        "P": bucket["P_sum"] / n,
        "recall": bucket["recall_sum"] / n,
        "precision": bucket["precision_sum"] / n,
    }


def fmt(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    return f"{100 * float(value):.2f}"


def aggregate_results(results_dir: Path) -> dict[str, Any]:
    per_type = defaultdict(new_bucket)
    overall = new_bucket()

    for path in load_result_files(results_dir):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        for row in iter_result_rows(payload):
            qtype = row.get("question_type") or path.stem
            add_row(per_type[qtype], row)
            add_row(overall, row)

    summary = {
        "results_dir": str(results_dir),
        "overall": finalize_bucket(overall),
        "per_type": {
            qtype: finalize_bucket(bucket)
            for qtype, bucket in sorted(per_type.items())
        },
    }
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print("\n===== PhysLogic Results =====")
    print("Type       N     Acc     F       O       P       Recall  Precision")
    for qtype, stats in summary["per_type"].items():
        print(
            f"{qtype:<10} "
            f"{stats['num_examples']:<5} "
            f"{fmt(stats['accuracy']):<7} "
            f"{fmt(stats['F']):<7} "
            f"{fmt(stats['O']):<7} "
            f"{fmt(stats['P']):<7} "
            f"{fmt(stats['recall']):<7} "
            f"{fmt(stats['precision']):<7}"
        )
    overall = summary["overall"]
    print(
        f"{'overall':<10} "
        f"{overall['num_examples']:<5} "
        f"{fmt(overall['accuracy']):<7} "
        f"{fmt(overall['F']):<7} "
        f"{fmt(overall['O']):<7} "
        f"{fmt(overall['P']):<7} "
        f"{fmt(overall['recall']):<7} "
        f"{fmt(overall['precision']):<7}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate PhysLogic benchmark results.")
    parser.add_argument("--results_dir", type=Path, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--model_id", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--save_json", action="store_true")
    args = parser.parse_args()

    if args.results_dir is not None:
        results_dir = args.results_dir
    elif args.run_name is not None:
        results_dir = DEFAULT_RESULTS_DIR / args.run_name
    elif args.model_id is not None:
        print("Warning: --model_id is deprecated; use --run_name or --results_dir.", flush=True)
        results_dir = DEFAULT_RESULTS_DIR / args.model_id
    else:
        raise ValueError("Provide --results_dir or --run_name")

    summary = aggregate_results(results_dir)
    print_summary(summary)
    if args.save_json:
        out_path = results_dir / "summary.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
