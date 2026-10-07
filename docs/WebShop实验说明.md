# ExpeL Local — Ollama / WebShop

基于 **ExpeL: LLM Agents Are Experiential Learners（AAAI 2024）** 的推理侧机制实现。使用本地 Ollama 的 `qwen3.5:9b`，模型参数全程冻结，通过任务轨迹提炼规则、检索成功案例，在独立任务上评测。

- 论文：https://arxiv.org/abs/2308.10144
- 作者代码：https://github.com/LeapLabTHU/ExpeL
- 参考提交：`e41ec9a24823e7b560c561ab191441b56d9bcefc`
- WebShop 环境：https://github.com/princeton-nlp/WebShop

## WebShop 运行

智能体访问真正的 WebShop 服务，执行商品搜索、打开商品、选择规格和模拟购买。产品、搜索索引和最终奖励由 WebShop 官方环境提供；本项目负责模型调用、动作适配和经验学习。

前提：Windows 的 Python 3.10+、已启动的 Ollama（模型 `qwen3.5:9b`），以及可访问的 WebShop 服务。模型端无需 OpenAI API Key，也无需额外 Python 包。商品数据和服务须另行安装，具体步骤见 [WebShop运行说明.md](WebShop运行说明.md)。

在项目根目录运行；默认服务地址为 `http://127.0.0.1:3000`：

```powershell
python -m expel webshop-check
python -m expel doctor --environment webshop
python -m expel one --environment webshop --out runs/webshop-one
python -m expel pipeline --environment webshop --out runs/webshop-pilot
```

`webshop-check` 只检查购物服务和任务对应关系，不调用模型。`doctor` 检查模型与环境连通性。`one` 在经验集侧运行一条购物任务；成功与否看 `metrics.success` 和 `metrics.reward`。`pipeline` 默认使用 8 条经验任务、4 条独立验证任务，完整执行经验收集、反思重试、规则提炼与四组验证。

扩大到 30 条经验任务、20 条验证任务：

```powershell
python -m expel pipeline --environment webshop --train-limit 30 --eval-limit 20 --out runs/webshop-30-20
```

连接其他机器上已搭好的服务时，每条命令都加 `--webshop-url http://服务器地址:3000`。启动服务的机器、商品库、任务文件、模型及步数预算应在对照组之间保持一致。

## 任务与验证划分

包内 `data/webshop_tasks.json` 来自作者代码中的 100 条 WebShop 任务，仅包含公开指令与会话索引，已移除目标商品和答案字段。2 条固定演示来自作者提示词。100 条是 ExpeL 任务子集，不能理解为整个 WebShop 只有 100 条任务。

WebShop 默认固定种子 42、两折划分，每折独立构建经验。经验集与验证集的任务 ID 不重叠。两折设置与作者配置一致，但此实现不宣称使用了作者完全相同的任务排列或完整评测协议。

不同商品库或服务版本可能使 `fixed_0` 对应不同指令。程序会检查当前任务的服务器指令，发现不一致即停止。可从正在使用的服务导出公开任务：

```powershell
python -m expel webshop-export --task-count 100 --export-path data/webshop_server_tasks.json
python -m expel pipeline --environment webshop --data data/webshop_server_tasks.json --out runs/webshop-server-pilot
```

导出文件不包含目标商品、目标规格或答案；后续各阶段使用相同任务文件。`--webshop-catalog small` 或 `full` 可记录商品库配置，该字段为实验者声明，程序不会自动验证商品总数。

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

WebShop 结果不使用 HotpotQA 的答案 EM/F1。此版本未预填任何真实 WebShop 成绩。

## 输出与恢复

每个实验目录保存 `manifest.json`（模型、配置与任务划分）、`experience.json`（经验轨迹）、`insights.json`（规则更新）、`evaluation.json`（逐任务动作与奖励）、`summary.csv` / `summary.json` 和 `report.md`。

完整报告只在所有验证任务与条件运行完毕后生成。运行中断后重跑完全相同的命令可恢复；更换模型、代码、任务、服务或实验参数时使用新的 `--out`。同一地址上的服务器数据变更无法仅通过 URL 检测，因此应另建实验目录。

BM25 为默认检索器；可选 `mpnet`：

```powershell
python -m pip install -r requirements-semantic.txt
python -m expel pipeline --environment webshop --retriever mpnet --out runs/webshop-mpnet
```

两折完整子集评测（每折重新收集经验，耗时高于试跑）：

```powershell
python tools/run_folds.py --environment webshop --out runs/webshop-folds
```

## 实现与测试

`expel/webshop.py` 实现网页观察、单步动作和官方奖励读取；`pipeline.py` 实现端到端流程；`memory.py` 提炼经验规则；`retrieval.py` 检索成功案例；`client.py` 调用真实 Ollama。命令行不会在服务失败时切换成模拟模型。

```powershell
python -m unittest discover -s tests -v
```

测试使用合成 HTTP 页面和脚本模型验证状态、评分、上下文隔离与断点恢复，不能作为真实模型的任务成绩。验证记录见 [VALIDATION.md](VALIDATION.md)。原有 HotpotQA 功能保留，说明见 [docs/HotpotQA运行说明.md](docs/HotpotQA运行说明.md)。
