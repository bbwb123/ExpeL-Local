# 基于 Qwen3.5-9B 的 ExpeL 推理侧机制复现

## 项目简介

本项目复现 **ExpeL: LLM Agents Are Experiential Learners（AAAI 2024）** 的推理侧核心机制。实验使用本地 Ollama 部署的 Qwen3.5-9B，模型参数全程冻结，不进行训练、微调或 LoRA 更新。

项目包含两个任务环境：

- **HotpotQA 多跳问答**：30 道经验题 + 20 道验证题，结果见 [report.md](report.md)（对应代码版本：标签 `hotpotqa-30x20`）。
- **WebShop 网页购物**（新增）：智能体访问真实的 WebShop 服务完成搜索、选商品、选规格和下单，10 道经验题 + 10 道验证题，结果见 [results/webshop_10x10](results/webshop_10x10)，排查与修改记录见 [CHANGES.md](CHANGES.md)。下文的 [WebShop 扩展](#webshop-扩展) 一节有摘要。

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
- `expel/webshop.py`：WebShop HTTP 适配、购物智能体提示词、动作解析与反思。
- `expel/client.py`：Ollama 调用、token 与耗时记录（思考模式只用于行动模型）。
- `probe.py` / `show.py`：小规模探针与单条轨迹查看。
- `tests/`：评分、数据隔离、工具状态、规则操作和流程恢复测试。

## WebShop 扩展

### 做了什么

- **环境接入**：通过 HTTP 适配官方 WebShop Flask 服务（WSL 中运行，small 商品库 + 官方自动生成的指令），观察只取页面可见文本，奖励只读官方结算页；`webshop-export` 从服务导出任务并逐条校验指令一致。
- **问题定位与修复**：最初智能体从不点进商品，在结果页反复搜索，reward 恒为 0。逐步对照轨迹后定位到三点：关闭思考时模型不读结果就重新搜索；提示词要求“完全符合才购买”导致从不下单；模型不知道剩余步数。对应修改：思考模式只用于行动模型；提示词说明部分得分、用 `< Prev` 回到同一结果页看下一个商品；每步附上可用动作和剩余步数。8 题探针上平均奖励由 0.25 提高到 0.64。
- **工程问题**：调用日志改为深拷贝（原日志会被后续对话改写）；反思/规则提炼不用思考模式（否则思考耗尽生成上限、输出为空）；单次生成失控时记为一步无效动作而不中断实验；上下文改为 32768；实验指纹记录采样设置；采纳审阅意见改写反思提示词（≤120 词、区分事实与推测、不编造筛选等不存在的操作）。
- **数据核查**：发现官方数据中有商品把 “Do not dry clean” 抽成了属性 `dry clean`，官方评分器也不处理否定；本实验 100 题中影响 3 题。按原样保留，不改任务和评分。

所有修改对四组（react / insights_only / retrieval_only / expel）同时生效；15 步上限、reward=1.0 才算成功均未改动。

### 10/10 结果（`results/webshop_10x10`）

| 条件 | 成功 | 平均奖励 | 未下单 |
|---|---:|---:|---:|
| react（基础组） | 3/10 | 0.383 | 6 |
| insights_only | 4/10 | 0.483 | 5 |
| retrieval_only | 1/10 | 0.298 | 6 |
| expel | 3/10 | 0.413 | 5 |

经验收集 10 题成功 7 题，形成 3 个成功/失败对比组，提炼出 3 条规则。完整 ExpeL 与基础组成功的题目完全相同，平均奖励的差别来自一题部分得分；样本量小，尚不能说明经验机制带来净提升。主要失败方式是搜索循环（40 次评测中 22 次未下单），其次是上述数据冲突和漏选规格。

### 运行

需要已启动的 WebShop 服务（安装见 [WebShop运行说明.md](WebShop运行说明.md)）和 Ollama（本次为 0.35.1）。在 PowerShell 中：

```powershell
$env:EXPEL_THINK = '1'; $env:EXPEL_TEMPERATURE = '0'
python -m expel webshop-export --task-count 100 --export-path data\webshop_server_tasks.json
python -m expel pipeline --environment webshop --data data\webshop_server_tasks.json --webshop-catalog small --num-predict 4096 --num-ctx 32768 --train-limit 10 --eval-limit 10 --out runs\webshop-10-10
```

WebShop 每次重启会重新随机各题的价格上限，重启后需重新导出任务文件。更多说明见 [docs/WebShop实验说明.md](docs/WebShop实验说明.md)。

## 当前结论

本项目已使用冻结的 Qwen3.5-9B 跑通 ExpeL 推理侧的经验收集、反思、规则提炼、成功案例检索和独立验证集评测。30/20 实验中，完整 ExpeL 与基础组的 EM 均为 0.600，经验机制改变了部分题目的结果，但没有形成净提升。后续可进一步扩大经验集和验证集，并分析规则质量与案例相关性对结果的影响。

WebShop 上同样跑通了完整流程（10/10）：完整 ExpeL 与基础组成功题目相同，尚无净提升；主要瓶颈是行动策略中的搜索循环。
