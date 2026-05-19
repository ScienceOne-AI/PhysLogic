# PhysLogic

[English](README.md) | **简体中文**

<p align="center">
  <a href="https://arxiv.org/pdf/2605.17104">
    <img src="https://img.shields.io/badge/arXiv-2605.17104-b31b1b?style=for-the-badge&logo=arxiv" alt="arXiv:2605.17104">
  </a>
  <a href="https://huggingface.co/datasets/ScienceOne-AI/PhysLogic">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow?style=for-the-badge" alt="Hugging Face Dataset">
  </a>
  <a href="https://github.com/ScienceOne-AI/PhysLogic">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github" alt="GitHub Repository">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-Apache--2.0-lightgrey?style=for-the-badge" alt="License: Apache-2.0">
  </a>
</p>

PhysLogic 是一个面向物理推理的 benchmark，用于同时评测模型的最终答案正确性和推理过程的逻辑性。本项目对应论文
*Scientific Logicality Enriched Methodology for LLM Reasoning: A Practice in Physics*。

本仓库只包含评测代码。benchmark 数据托管在 Hugging Face，并在运行时加载。

🎉 本工作已被 ICML 2026 接收。

## 亮点

- **面向推理过程的评测**：PhysLogic 不只判断模型是否得到正确答案，还评估其推理过程是否遵循问题背后的科学逻辑。
- **来源于物理论文的问题**：问题构造基于物理学术论文中的核心逻辑推导，而不是孤立的教材习题。
- **逻辑性标注**：每道题都包含有序的 logical nexuses 及其重要性权重，可用于自动评测推理内容覆盖、顺序一致性和前向推进程度。
- **结构化的 benchmark 划分**：公开 benchmark 覆盖物理子领域、难度层级和题型等多个维度。

## Benchmark 思路

PhysLogic 从两个角度评测物理问题求解：最终答案正确性，以及推理过程的逻辑性。benchmark 使用 **scientific logicality** 这一概念：模型回答会与最终答案以及解题所需的关键逻辑步骤进行比较。

评测器会将模型推理过程与 logical nexuses 对齐，并计算三个指标：

- `F`: Logical Fidelity，衡量模型是否覆盖必要的逻辑内容；
- `O`: Causal Connection，衡量模型是否保持预期推导顺序；
- `P`: Inferential Progress，衡量推理路径是否持续向前推进。

## 安装

建议使用 Python 3.10+。

```bash
pip install -r requirements.txt
```

设置 OpenAI-compatible Chat Completions endpoint 的 API key：

```bash
export OPENAI_API_KEY=...
```

如果使用非 OpenAI 服务，可以通过 `--base_url` 指定 endpoint，并通过
`--api_key_env` 指定 API key 所在的环境变量。

## 快速开始

### 评测 API 模型

```bash
python src/benchmarking.py \
  --model_id gpt-4o-mini \
  --run_name gpt-4o-mini \
  --concurrency 12
```

`--concurrency` 表示并发 API worker 数，不是单次请求中的 batch size。请根据服务商的 rate limit 调整该参数。

如果使用本地或第三方 OpenAI-compatible endpoint：

```bash
python src/benchmarking.py \
  --model_id your-model \
  --base_url http://localhost:8000/v1 \
  --api_key_env OPENAI_API_KEY \
  --run_name your-model \
  --concurrency 8
```

结果会写入：

```text
results/<run_name>/{choice,comp_n,comp_e,proof}.json
```

### 评测已有模型输出

如果你已经生成了模型回答，可以准备一个 JSONL 文件，每行一个样本：

```json
{"uid": "example-id", "answer_pred": "solution text with \\boxed{...}", "reasoning_pred": "optional reasoning text"}
```

然后运行：

```bash
python src/benchmarking.py \
  --predictions_path predictions.jsonl \
  --run_name my_predictions
```

`reasoning_pred` 是可选字段。如果没有提供，评测脚本会优先使用 API 返回中的
`reasoning_content`；如果没有，则尝试使用 `<think>...</think>` 中的内容；仍然没有时，会使用完整可见输出作为 reasoning。

### 汇总结果

```bash
python src/result_summary.py --run_name gpt-4o-mini --save_json
```

该命令会输出每类题型和 overall 的平均结果：

- `Acc`: 只在 `choice` 和 `comp_n` 上统计的 macro accuracy
- `F/O/P`: 所有已评测样本上的 macro average
- `Recall/Precision`: Logical Fidelity 的辅助统计

也可以直接指定结果目录：

```bash
python src/result_summary.py --results_dir results/gpt-4o-mini
```

## 代码结构

```text
src/benchmarking.py          benchmark 主入口和 CLI 编排
src/model_client.py          OpenAI-compatible API 并发调用客户端
src/answer_scoring.py        choice 和 comp_n 的最终答案正确率计算
src/logicality_metrics.py    F/O/P 逻辑性指标实现
src/result_summary.py        按题型和 overall 汇总结果
src/prompt/LLM_judge.md      comp_n 非数值可判定场景下的可选 LLM judge prompt
```

## 常用参数

```text
--output_dir results
--question_types choice,comp_n,comp_e,proof
--encoder_model all-MiniLM-L6-v2
--similarity_threshold 0.3
--judge_model_id <model>
--judge_concurrency 10
--limit_per_type 2
```

`--judge_model_id` 只在 `comp_n` 的答案无法通过数值抽取直接判断、需要 LLM 做文本判分时使用。
