import re
from pathlib import Path
from typing import Any


NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"


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
    pattern = (
        f"({NUMBER_PATTERN})"
        r"\s*(?:\\+times|\\+cdot|×|\*)\s*10\s*\^\s*\{?\s*"
        r"([-+]?\d+)"
        r"\s*\}?"
    )
    match = re.search(pattern, text.strip())
    if not match:
        return None
    coefficient = float(match.group(1))
    exponent = int(match.group(2))
    if exponent > 308 or exponent < -308:
        return None
    return coefficient * (10**exponent)


def power_of_ten_to_float(text: str) -> float | None:
    pattern = r"(?<![\d.])10\s*\^\s*\{?\s*([-+]?\d+)\s*\}?"
    match = re.search(pattern, text.strip())
    if not match:
        return None
    exponent = int(match.group(1))
    if exponent > 308 or exponent < -308:
        return None
    return 10**exponent


def fraction_to_float(text: str) -> float | None:
    latex_pattern = (
        r"\\+(?:dfrac|tfrac|frac)\s*\{\s*"
        f"({NUMBER_PATTERN})"
        r"\s*\}\s*\{\s*"
        f"({NUMBER_PATTERN})"
        r"\s*\}"
    )
    plain_pattern = f"({NUMBER_PATTERN})" r"\s*/\s*" f"({NUMBER_PATTERN})"
    for pattern in (latex_pattern, plain_pattern):
        match = re.search(pattern, text.strip())
        if not match:
            continue
        numerator = float(match.group(1))
        denominator = float(match.group(2))
        if denominator == 0:
            return None
        return numerator / denominator
    return None


def extract_number(text: str) -> float | None:
    pattern = rf"{NUMBER_PATTERN}(?:[eE][+-]?\d+)?"
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
    value = power_of_ten_to_float(content)
    if value is not None:
        return value
    value = fraction_to_float(content)
    if value is not None:
        return value
    return extract_number(content)


def extract_choice_label(text: str | None) -> str | None:
    if text is None:
        return None
    match = re.search(r"\b([ABCD])\b", text.strip())
    if match:
        return match.group(1)
    match = re.search(r"[ABCD]", text.strip())
    return match.group(0) if match else None


def score_choice_answer(gold_answer: str, prediction: str | None) -> dict[str, Any]:
    extracted = extract_from_boxed(prediction)
    choice = extract_choice_label(extracted)
    gold_extracted = extract_from_boxed(gold_answer) or gold_answer
    gold_choice = extract_choice_label(gold_extracted)
    return {
        "answer_gt": gold_answer,
        "answer_pred": prediction,
        "answer_pred_extracted": extracted,
        "gold_choice": gold_choice,
        "extracted_choice": choice,
        "correct": 1 if choice is not None and choice == gold_choice else 0,
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
        correct = int(abs(gt_num - pred_num) <= 0.05 * abs(gt_num) + 1e-12)

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
    verdict_match = re.search(r"\b([AB])\b", answer.strip().upper())
    verdict = verdict_match.group(1) if verdict_match else None
    return {
        "answer_gt": gold_answer,
        "answer_pred": prediction,
        "answer_pred_extracted": extracted,
        "correct": 1 if verdict == "A" else 0,
        "judge_method": "llm_judge",
        "judge_verdict": verdict,
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
