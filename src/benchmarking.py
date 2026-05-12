import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from answer_scoring import score_answer
from model_client import OpenAICompatibleClient


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HF_DATASET = "<HF_DATASET_REPO_ID>"
DEFAULT_HF_CONFIG = "default"
DEFAULT_HF_SPLIT = "test"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results"
SUPPORTED_QUESTION_TYPES = ("choice", "comp_n", "comp_e", "proof")


def parse_question_types(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(SUPPORTED_QUESTION_TYPES)
    question_types = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(question_types) - set(SUPPORTED_QUESTION_TYPES))
    if unknown:
        raise ValueError(f"Unknown question types: {unknown}")
    return question_types


def validate_dataset(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    required = {
        "uid",
        "question",
        "final_answer",
        "logical_nexuses",
        "logical_nexus_weights",
        "question_type",
    }
    for idx, item in enumerate(data):
        missing = required - set(item)
        if missing:
            raise ValueError(f"Dataset item {idx} is missing fields: {sorted(missing)}")
        if len(item["logical_nexuses"]) != len(item["logical_nexus_weights"]):
            raise ValueError(f"Dataset item {item['uid']} has mismatched nexus weights")
    return data


def load_physlogic_dataset() -> list[dict[str, Any]]:
    if DEFAULT_HF_DATASET == "<HF_DATASET_REPO_ID>":
        raise ValueError(
            "The default HF dataset id is still a placeholder. "
            "Replace DEFAULT_HF_DATASET with the released Hugging Face dataset id."
        )
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency datasets. Install with: pip install -r requirements.txt"
        ) from exc
    dataset = load_dataset(DEFAULT_HF_DATASET, name=DEFAULT_HF_CONFIG, split=DEFAULT_HF_SPLIT)
    return validate_dataset([dict(row) for row in dataset])


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "uid" not in item or "answer_pred" not in item:
                raise ValueError(f"Prediction line {line_no} must contain uid and answer_pred")
            predictions[str(item["uid"])] = item
    return predictions


def load_sentence_encoder(model_name: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency sentence-transformers. Install with: pip install -r requirements.txt"
        ) from exc
    return SentenceTransformer(model_name)


def load_logicality_metric_fn() -> Any:
    try:
        from logicality_metrics import compute_logicality_metrics
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing metric dependencies. Install with: pip install -r requirements.txt"
        ) from exc
    return compute_logicality_metrics


def build_messages(question: str) -> list[dict[str, str]]:
    prompt = (
        f"{question}\n\n"
        "Please solve the problem step by step. Put the final answer at the end "
        "using \\boxed{}."
    )
    return [
        {"role": "system", "content": "You are a helpful assistant for physics problem solving."},
        {"role": "user", "content": prompt},
    ]


def infer_with_api(
    client: OpenAICompatibleClient,
    dataset: list[dict[str, Any]],
    concurrency: int,
) -> dict[str, dict[str, Any]]:
    messages = [build_messages(item["question"]) for item in dataset]
    responses = client.batch_generate(messages, concurrency=concurrency)
    return {
        str(item["uid"]): {
            "uid": item["uid"],
            "answer_pred": response.get("answer"),
            "reasoning_pred": response.get("reasoning"),
            "api_response": response.get("api_response"),
        }
        for item, response in zip(dataset, responses)
    }


def extract_reasoning(prediction: dict[str, Any]) -> str | None:
    explicit_reasoning = prediction.get("reasoning_pred")
    if explicit_reasoning:
        return str(explicit_reasoning).strip()

    answer = prediction.get("answer_pred")
    if answer is None:
        return None
    answer = str(answer)

    think_match = re.search(r"<think>(.*?)</think>", answer, flags=re.DOTALL | re.IGNORECASE)
    if think_match:
        return think_match.group(1).strip()
    if "</think>" in answer:
        return answer.split("</think>")[-1].strip()
    return answer.strip()


def judge_items(
    dataset: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    judge_client: OpenAICompatibleClient | None,
    judge_concurrency: int,
) -> dict[str, dict[str, Any] | None]:
    if judge_concurrency < 1:
        raise ValueError("judge_concurrency must be >= 1")

    def _judge(item: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        uid = str(item["uid"])
        prediction = predictions[uid].get("answer_pred")
        result = score_answer(
            question_type=item["question_type"],
            question=item["question"],
            gold_answer=item["final_answer"],
            prediction=prediction,
            judge_client=judge_client,
        )
        return uid, result

    results: dict[str, dict[str, Any] | None] = {}
    with ThreadPoolExecutor(max_workers=judge_concurrency) as pool:
        futures = [pool.submit(_judge, item) for item in dataset]
        for future in as_completed(futures):
            uid, result = future.result()
            results[uid] = result
    return results


def evaluate_logicality(
    dataset: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    encoder_model: Any,
    compute_logicality_metrics: Any,
    threshold: float,
) -> dict[str, dict[str, Any]]:
    results = {}
    for item in dataset:
        uid = str(item["uid"])
        reasoning = extract_reasoning(predictions[uid])
        results[uid] = compute_logicality_metrics(
            logical_nexuses=item["logical_nexuses"],
            logical_nexus_weights=item["logical_nexus_weights"],
            reasoning=reasoning,
            encoder_model=encoder_model,
            threshold=threshold,
        )
    return results


def build_sample_results(
    dataset: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    judge_results: dict[str, dict[str, Any] | None],
    logicality_results: dict[str, dict[str, Any]],
    save_api_responses: bool,
    save_similarity_matrix: bool,
) -> list[dict[str, Any]]:
    sample_results = []
    for item in dataset:
        uid = str(item["uid"])
        prediction = predictions[uid]
        logicality_result = dict(logicality_results[uid])
        if not save_similarity_matrix:
            logicality_result.pop("similarity_matrix", None)
            logicality_result.pop("M", None)
        result = {
            "uid": item["uid"],
            "question_type": item["question_type"],
            "difficulty": item.get("difficulty"),
            "subdomain": item.get("subdomain"),
            "answer_pred": prediction.get("answer_pred"),
            "reasoning_used": extract_reasoning(prediction),
            "judge_result": judge_results.get(uid),
            "logicality_result": logicality_result,
        }
        if save_api_responses and "api_response" in prediction:
            result["api_response"] = prediction["api_response"]
        sample_results.append(result)
    return sample_results


def default_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    if args.predictions_path:
        return Path(args.predictions_path).stem
    if args.model_id:
        return args.model_id.replace("/", "__")
    return "physlogic_run"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PhysLogic benchmark.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--api_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--judge_model_id", type=str, default=None)
    parser.add_argument("--judge_base_url", type=str, default=None)
    parser.add_argument("--judge_api_key_env", type=str, default=None)
    parser.add_argument("--judge_concurrency", type=int, default=10)
    parser.add_argument("--question_types", type=str, default="choice,comp_n,comp_e,proof")
    parser.add_argument("--predictions_path", type=Path, default=None)
    parser.add_argument("--encoder_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--similarity_threshold", type=float, default=0.3)
    parser.add_argument("--limit_per_type", type=int, default=None)
    parser.add_argument("--save_api_responses", action="store_true")
    parser.add_argument("--save_similarity_matrix", action="store_true")
    args = parser.parse_args()

    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")

    question_types = parse_question_types(args.question_types)
    dataset = load_physlogic_dataset()
    data_source = {
        "type": "huggingface",
        "hf_dataset": DEFAULT_HF_DATASET,
        "hf_config": DEFAULT_HF_CONFIG,
        "hf_split": DEFAULT_HF_SPLIT,
    }
    dataset = [item for item in dataset if item["question_type"] in question_types]
    if args.limit_per_type is not None:
        limited = []
        for qtype in question_types:
            limited.extend([item for item in dataset if item["question_type"] == qtype][: args.limit_per_type])
        dataset = limited

    print(f"Loading sentence encoder: {args.encoder_model}", flush=True)
    compute_logicality_metrics = load_logicality_metric_fn()
    encoder_model = load_sentence_encoder(args.encoder_model)

    if args.predictions_path:
        predictions = load_predictions(args.predictions_path)
        missing = [str(item["uid"]) for item in dataset if str(item["uid"]) not in predictions]
        if missing:
            raise ValueError(f"Predictions file is missing {len(missing)} uids, first missing uid: {missing[0]}")
    else:
        if not args.model_id:
            raise ValueError("--model_id is required unless --predictions_path is provided")
        client = OpenAICompatibleClient(
            model_id=args.model_id,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
        )
        predictions = infer_with_api(client, dataset, concurrency=args.concurrency)

    judge_model_id = args.judge_model_id
    if judge_model_id is None and not args.predictions_path:
        judge_model_id = args.model_id
    judge_client = None
    if judge_model_id:
        judge_client = OpenAICompatibleClient(
            model_id=judge_model_id,
            api_key_env=args.judge_api_key_env or args.api_key_env,
            base_url=args.judge_base_url or args.base_url,
            temperature=0.0,
        )

    run_name = default_run_name(args)
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "run_name": run_name,
        "model_id": args.model_id,
        "data_source": data_source,
        "encoder_model": args.encoder_model,
        "similarity_threshold": args.similarity_threshold,
        "concurrency": args.concurrency,
        "judge_model_id": judge_model_id,
        "judge_concurrency": args.judge_concurrency,
        "question_types": question_types,
    }

    for qtype in question_types:
        subset = [item for item in dataset if item["question_type"] == qtype]
        if not subset:
            continue
        print(f"Evaluating {qtype}: {len(subset)} examples", flush=True)
        judge_results = judge_items(
            dataset=subset,
            predictions=predictions,
            judge_client=judge_client,
            judge_concurrency=args.judge_concurrency,
        )
        logicality_results = evaluate_logicality(
            dataset=subset,
            predictions=predictions,
            encoder_model=encoder_model,
            compute_logicality_metrics=compute_logicality_metrics,
            threshold=args.similarity_threshold,
        )
        payload = {
            "metadata": metadata | {"question_type": qtype},
            "results": build_sample_results(
                dataset=subset,
                predictions=predictions,
                judge_results=judge_results,
                logicality_results=logicality_results,
                save_api_responses=args.save_api_responses,
                save_similarity_matrix=args.save_similarity_matrix,
            ),
        }
        out_path = run_dir / f"{qtype}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
