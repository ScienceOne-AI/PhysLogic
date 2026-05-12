import re
from pathlib import Path
from typing import Any


def extract_from_boxed(answer: str | None) -> str | None:
    """Extract the final content from the last LaTeX \\boxed{...} block."""
    if answer is None:
        return None
    last_boxed_start = answer.rfind("\\boxed{")
    if last_boxed_start == -1:
        return None

    content_start = last_boxed_start + len("\\boxed{")
    brace_count = 1
    for i in range(content_start, len(answer)):
        if answer[i] == "{":
            brace_count += 1
        elif answer[i] == "}":
            brace_count -= 1
        if brace_count == 0:
            return answer[content_start:i].strip()
    return answer[content_start:].strip()


def scientific_notation_to_float(text: str) -> float | None:
    pattern = r"(-?\d+(?:\.\d+)?)\s*\\times\s*10\^\{(-?\d+)\}"
    match = re.search(pattern, text.strip())
    if not match:
        return None
    coefficient = float(match.group(1))
    exponent = int(match.group(2))
    if exponent > 308 or exponent < -308:
        return None
    return coefficient * (10**exponent)


def extract_number(text: str) -> float | None:
    pattern = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
    match = re.search(pattern, text.strip())
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def get_final_math_answer(content: str | None) -> float | None:
    if content is None:
        return None
    value = scientific_notation_to_float(content)
    if value is not None:
        return value
    return extract_number(content)


def score_choice_answer(gold_answer: str, prediction: str | None) -> dict[str, Any]:
    extracted = extract_from_boxed(prediction)
    choice = None
    if extracted:
        match = re.search(r"[ABCD]", extracted)
        choice = match.group(0) if match else None
    return {
        "answer_gt": gold_answer,
        "answer_pred": prediction,
        "answer_pred_extracted": extracted,
        "extracted_choice": choice,
        "correct": 1 if choice == gold_answer else 0,
        "judge_method": "choice_exact",
    }


def score_numeric_answer_by_value(gold_answer: str, prediction: str | None) -> dict[str, Any] | None:
    extracted = extract_from_boxed(prediction)
    if extracted is None:
        return {
            "answer_gt": gold_answer,
            "answer_pred": prediction,
            "answer_pred_extracted": None,
            "correct": 0,
            "judge_method": "missing_boxed_answer",
        }

    gt_num = get_final_math_answer(gold_answer)
    pred_num = get_final_math_answer(extracted)
    if gt_num is None or pred_num is None:
        return None

    if gt_num == 0:
        correct = int(abs(pred_num) < 1e-3)
    else:
        correct = int(abs(gt_num - pred_num) / abs(gt_num) <= 0.05)

    return {
        "answer_gt": gold_answer,
        "answer_pred": prediction,
        "answer_pred_extracted": extracted,
        "gold_numeric": gt_num,
        "pred_numeric": pred_num,
        "correct": correct,
        "judge_method": "numeric_tolerance",
    }


def score_numeric_or_text_answer(
    question: str,
    gold_answer: str,
    prediction: str | None,
    judge_client: Any | None = None,
    judge_prompt_path: Path | None = None,
) -> dict[str, Any]:
    numeric_result = score_numeric_answer_by_value(gold_answer, prediction)
    if numeric_result is not None:
        return numeric_result

    extracted = extract_from_boxed(prediction)
    if judge_client is None:
        return {
            "answer_gt": gold_answer,
            "answer_pred": prediction,
            "answer_pred_extracted": extracted,
            "correct": 0,
            "judge_method": "unjudged_non_numeric",
            "judge_error": "Non-numeric answer requires --judge_model_id for LLM judging.",
        }

    if judge_prompt_path is None:
        judge_prompt_path = Path(__file__).resolve().parent / "prompt" / "LLM_judge.md"
    judge_prompt = judge_prompt_path.read_text(encoding="utf-8")
    user_prompt = (
        judge_prompt.replace("{question}", question)
        .replace("{pred}", extracted or "no answer")
        .replace("{gold}", gold_answer)
    )
    response = judge_client.batch_generate(
        [[
            {"role": "system", "content": "You are a helpful assistant that judges answer correctness."},
            {"role": "user", "content": user_prompt},
        ]],
        concurrency=1,
    )[0]
    answer = response.get("answer") or ""
    return {
        "answer_gt": gold_answer,
        "answer_pred": prediction,
        "answer_pred_extracted": extracted,
        "correct": 1 if "A" in answer else 0,
        "judge_method": "llm_judge",
        "judge_response": response,
    }


def score_answer(
    question_type: str,
    question: str,
    gold_answer: str,
    prediction: str | None,
    judge_client: Any | None = None,
) -> dict[str, Any] | None:
    """Return answer-accuracy judgement, or None for types without accuracy."""
    if question_type == "choice":
        return score_choice_answer(gold_answer, prediction)
    if question_type == "comp_n":
        return score_numeric_or_text_answer(question, gold_answer, prediction, judge_client)
    return None
