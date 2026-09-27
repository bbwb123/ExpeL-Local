# 基于 Qwen3.5-9B 的 ExpeL 推理侧机制复现

## 项目简介

本项目复现 **ExpeL: LLM Agents Are Experiential Learners（AAAI 2024）** 的推理侧核心机制。实验使用本地 Ollama 部署的 Qwen3.5-9B，模型参数全程冻结，不进行训练、微调或 LoRA 更新。

- 论文：[ExpeL: LLM Agents Are Experiential Learners](https://arxiv.org/abs/2308.10144)
- 官方代码：[LeapLabTHU/ExpeL](https://github.com/LeapLabTHU/ExpeL)
- 参考提交：`e41ec9a24823e7b560c561ab191441b56d9bcefc`
- 实验模型：`qwen3.5:9b`（9.7B，Q4_K_M）
- 运行方式：Windows + Ollama 0.34.4

## 复现内容

项目实现了以下流程：

1. 使用工具调用智能体完成多步问答任务，循环执行思考、搜索、观察和作答。
2. 对经验集任务进行多次尝试，并对失败轨迹生成反思。
3. 从成功与失败轨迹的对比中提炼可复用经验规则。
4. 按问题相似度检索成功案例，将相关轨迹作为示例加入后续推理。
5. 在独立验证集上分别测试基础智能体、仅规则、仅案例检索和完整 ExpeL。
6. 输出 EM、F1、平均推理步数、运行时间及逐题变化情况。

模型在全部阶段保持冻结。经验只以文本规则和成功轨迹的形式加入上下文，不修改模型权重。

## 数据与实验设置

实验使用 ExpeL 官方仓库附带的 100 道 HotpotQA 样本，并固定划分经验集和验证集，避免验证题进入经验规则或案例检索池。

本次实验配置：

| 配置项 | 设置 |
|---|---|
| 经验样例 | 30 道 |
| 验证样例 | 20 道 |
| 每道经验题最大尝试次数 | 3 次（1 次初始尝试 + 2 次反思重试） |
| 每轮最大行动步数 | 7 |
| 检索方法 | BM25 |
| 经验规则数量 | 2 条 |
| 成功/失败对比组 | 2 组 |
| 成功轨迹批次 | 8 组 |
| 模型参数更新 | 无 |

默认环境将样本附带的百科段落合并为固定检索库，用于验证经验学习机制和端到端流程。该设置不等同于论文中的在线 Wikipedia 全站搜索。

## 运行方法

### 1. 环境准备

安装 Python 3.10 或更高版本、Windows 版 Ollama，并下载模型：

```powershell
ollama pull qwen3.5:9b
```

### 2. 验证模型连接

```powershell
python -m expel doctor
```

程序会实际调用本地模型，并检查是否返回 `LOCAL_MODEL_OK`。

### 3. 运行单题

```powershell
python -m expel one
```

### 4. 复现实验配置

```powershell
python -m expel pipeline --train-limit 30 --eval-limit 20 --out runs/expel-30-20
```

运行中会按任务保存检查点；中断后使用完全相同的命令可以继续。改变模型、代码或实验参数时，应使用新的输出目录，避免混合不同实验结果。

## 主要输出

| 文件 | 内容 |
|---|---|
| `manifest.json` | 模型信息、实验参数、代码与数据指纹、数据划分 |
| `experience.json` | 经验任务的多次尝试、反思和成功/失败轨迹 |
| `insights.json` | 提炼出的经验规则及其来源 |
| `evaluation.json` | 四种条件的逐题推理轨迹与评分 |
| `summary.csv` / `summary.json` | 汇总指标与逐题变化统计 |
| `report.md` | 自动生成的中文实验报告 |

## 代码结构

- `expel/agent.py`：多步工具调用与行动解析。
- `expel/pipeline.py`：经验收集、规则提炼、验证和结果汇总。
- `expel/memory.py`：轨迹分组、规则生成和规则票数更新。
- `expel/retrieval.py`：成功案例检索。
- `expel/environment.py`：固定百科语料库与搜索工具。
- `expel/client.py`：Ollama 调用、token 与耗时记录。
- `tests/`：评分、数据隔离、工具状态、规则操作和流程恢复测试。

## 当前结论

本项目已使用冻结的 Qwen3.5-9B 跑通 ExpeL 推理侧的经验收集、反思、规则提炼、成功案例检索和独立验证集评测。30/20 实验中，完整 ExpeL 与基础组的 EM 均为 0.600，经验机制改变了部分题目的结果，但没有形成净提升。后续可进一步扩大经验集和验证集，并分析规则质量与案例相关性对结果的影响。
