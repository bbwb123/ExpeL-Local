# ExpeL 本地复现

基于 **ExpeL: LLM Agents Are Experiential Learners（AAAI 2024）** 的推理侧机制实现。使用本地 Ollama 的 `qwen3.5:9b`，模型参数全程冻结，通过任务轨迹提炼规则、检索成功案例，在独立任务上评测。

- 论文：https://arxiv.org/abs/2308.10144
- 作者代码：https://github.com/LeapLabTHU/ExpeL
- 参考提交：`e41ec9a24823e7b560c561ab191441b56d9bcefc`
- WebShop 环境：https://github.com/princeton-nlp/WebShop

项目包含两部分实验：

- **HotpotQA 多跳问答**：30 条经验任务 + 20 条验证任务，结果见 [report.md](report.md)（对应代码版本：标签 `hotpotqa-30x20`）。
- **WebShop 网页购物**：10 条经验任务 + 10 条验证任务。

## 任务与验证划分

本轮 WebShop 实验使用 small 商品库（1000 件商品）和合成任务指令。任务从当前运行的 WebShop 服务导出：

```powershell
python -m expel webshop-export --task-count 100 --export-path data/webshop_server_tasks.json
```

导出的任务按固定种子 42 打乱后做两折划分（取第 0 折），经验集和验证集各 50 条、互不重叠；本次从中各取前 10 条，即 10 条经验任务 + 10 条验证任务。2 条固定演示来自作者提示词。

## 四组对照与评分

| 条件 | 经验规则 | 成功案例检索 |
|---|---|---|
| `react` | 关闭 | 关闭，使用固定演示 |
| `insights_only` | 启用 | 关闭，使用固定演示 |
| `retrieval_only` | 关闭 | 启用 |
| `expel` | 启用 | 启用 |

所有条件使用同一模型、工具协议和最大 15 步行动预算，演示数量相同。每次尝试使用独立购物会话。验证任务不参与规则提炼和检索案例池构建，验证阶段经验保持冻结。

- **成功率**：购买后官方环境奖励等于 1.0 的任务比例。
- **平均奖励**：官方环境奖励的均值，保留部分满足要求的得分。
- 耗尽步数而未购买的任务记 0 分，同时记录 `max_steps` 状态。

## 输出与恢复

每个实验目录保存 `manifest.json`（模型、配置与任务划分）、`experience.json`（经验轨迹）、`insights.json`（规则更新）、`evaluation.json`（逐任务动作与奖励）、`summary.csv` / `summary.json` 和 `report.md`。

BM25 为默认检索器；可选 `mpnet`：

```powershell
python -m pip install -r requirements-semantic.txt
python -m expel pipeline --environment webshop --retriever mpnet --out runs/webshop-mpnet
```

## 实现与测试

`expel/webshop.py` 实现网页观察、单步动作和官方奖励读取；`pipeline.py` 实现端到端流程；`memory.py` 提炼经验规则；`retrieval.py` 检索成功案例；`client.py` 调用真实 Ollama。

```powershell
python -m unittest discover -s tests -v
```

## 本轮代码修改

- `expel/webshop.py`：修改购物与导航提示词，加入可用动作和剩余步数；精简反思提示词；处理格式错误与生成截断。
- `expel/client.py`：配置思考模式与温度，思考仅用于 actor；调用日志保存深拷贝；新增生成截断异常。
- `expel/pipeline.py`：将采样配置和思考作用范围记录到 manifest 与实验指纹。
- `tests/test_core.py`、`tests/test_webshop.py`：补充日志快照、思考作用范围和截断处理测试。
- `probe.py`、`show.py`：提供配置对比与轨迹查看工具。

四组共用同一 actor、模型及15步预算，模型参数冻结。检索组使用成功案例替换固定示范。

实验结果见 [WebShop 实验报告](results/webshop_10x10/report.md)，排查与修改的完整记录见 [CHANGES.md](CHANGES.md)。
