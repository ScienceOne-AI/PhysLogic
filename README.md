# PhysLogic

**English** | [简体中文](README_zh.md)

<p align="center">
  <a href="https://huggingface.co/datasets/ScienceOne-AI/PhysLogic">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow?style=for-the-badge" alt="Hugging Face Dataset">
  </a>
  <a href="https://github.com/ScienceOne-AI/PhysLogic">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github" alt="GitHub Repository">
  </a>
  <a href="https://creativecommons.org/licenses/by-nc/4.0/">
    <img src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey?style=for-the-badge" alt="License: CC BY-NC 4.0">
  </a>
</p>

PhysLogic is a physics reasoning benchmark for evaluating both final-answer
accuracy and the logicality of model reasoning processes.

This repository contains the evaluation code. The benchmark data are hosted on
Hugging Face and loaded at runtime.

🎉 This work has been accepted to ICML 2026.

## Highlights

- **Process-aware evaluation**: PhysLogic evaluates not only whether a model
  reaches the right answer, but also whether its reasoning follows the core
  scientific logic behind the problem.
- **Paper-derived physics problems**: questions are constructed from logical
  derivations in physics academic papers, rather than from isolated textbook
  exercises.
- **Logicality annotations**: each example includes ordered logical nexuses and
  their importance weights, enabling automatic evaluation of reasoning fidelity,
  ordering, and progress.
- **Structured benchmark split**: the released benchmark covers physics
  subdomains, difficulty levels, and question types in a structured way.

## Benchmark Design

PhysLogic evaluates physics problem solving from two views: final-answer
accuracy and reasoning-process logicality. The benchmark uses the concept of
**scientific logicality**: a model response is compared with the key logical
steps needed to solve the problem and with the final answer.

The evaluator compares model reasoning against the logical nexuses using three
metrics:

- `F`: Logical Fidelity, measuring coverage of required logical content;
- `O`: Causal Connection, measuring consistency with the intended derivational
  order;
- `P`: Inferential Progress, measuring forward movement in the reasoning path.

## Installation

Use Python 3.10+.

```bash
pip install -r requirements.txt
```

Set an API key for an OpenAI-compatible Chat Completions endpoint:

```bash
export OPENAI_API_KEY=...
```

For non-OpenAI providers, pass `--base_url` and optionally choose a different
API-key environment variable with `--api_key_env`.

## Quick Start

### Evaluate An API Model

```bash
python src/benchmarking.py \
  --model_id gpt-4o-mini \
  --run_name gpt-4o-mini \
  --concurrency 12
```

`--concurrency` is the number of parallel API workers. It is not a single-request
batch size. Increase it only within your provider's rate limits.

For an OpenAI-compatible local or third-party endpoint:

```bash
python src/benchmarking.py \
  --model_id your-model \
  --base_url http://localhost:8000/v1 \
  --api_key_env OPENAI_API_KEY \
  --run_name your-model \
  --concurrency 8
```

Outputs are written to:

```text
results/<run_name>/{choice,comp_n,comp_e,proof}.json
```

### Evaluate Existing Predictions

If you already generated model outputs, create a JSONL file with one object per
line:

```json
{"uid": "example-id", "answer_pred": "solution text with \\boxed{...}", "reasoning_pred": "optional reasoning text"}
```

Then run:

```bash
python src/benchmarking.py \
  --predictions_path predictions.jsonl \
  --run_name my_predictions
```

`reasoning_pred` is optional. If it is absent, the evaluator uses API
`reasoning_content` when available, then `<think>...</think>` content when
present, and otherwise the full visible output.

### Summarize Results

```bash
python src/result_summary.py --run_name gpt-4o-mini --save_json
```

This prints per-type and overall averages:

- `Acc`: macro accuracy over `choice` and `comp_n` only
- `F/O/P`: macro averages over all evaluated examples
- `Recall/Precision`: supporting values for Logical Fidelity

For direct aggregation from a path:

```bash
python src/result_summary.py --results_dir results/gpt-4o-mini
```

## Code Structure

```text
src/benchmarking.py          Benchmark runner and CLI orchestration
src/model_client.py          OpenAI-compatible API client with concurrent requests
src/answer_scoring.py        Final-answer accuracy scoring for choice and comp_n
src/logicality_metrics.py    F/O/P logicality metric implementation
src/result_summary.py        Per-type and overall result aggregation
src/prompt/LLM_judge.md      Optional LLM judge prompt for non-numeric comp_n cases
```

## Useful Options

```text
--output_dir results
--question_types choice,comp_n,comp_e,proof
--encoder_model all-MiniLM-L6-v2
--similarity_threshold 0.3
--judge_model_id <model>
--judge_concurrency 10
--limit_per_type 2
```

`--judge_model_id` is only needed when a `comp_n` answer cannot be judged by
numeric extraction and requires an LLM textual judgement.
