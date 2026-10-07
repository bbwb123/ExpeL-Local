# ExpeL Local — 冻结模型的经验学习复现

> v0.1.1：严格限制每轮只能产生一个动作；格式不合格的模型输出仅写入轨迹日志，
> 不再回填模型上下文，避免模型自造的 Observation 污染后续推理。

> v0.1.2：明确搜索结果里的准确标题应通过 `Search[标题]` 打开，
> `Lookup` 仅用于已打开页面内的关键词查找；直接证据充分时及时结束。
> 这属于所有对照组共享的工具协议调整，尚未证明会提高验证集指标。

基于 **ExpeL: LLM Agents Are Experiential Learners（AAAI 2024）** 的本地推理侧实现。
目标：使用 Ollama 上的 Qwen3.5-9B，完成经验收集、经验规则提炼、成功案例检索和独立任务评测。

**当前交付状态：代码与离线流程测试通过。用户在 Windows 本地完成的 v0.1.1 实验（30 个经验样例、20 个验证样例）中，基础组与完整 ExpeL 的 EM 均为 0.60；v0.1.2 的 8/4 试跑四组 EM 均为 0.50，证明新版流程可运行，但尚未完成新版 30/20 评测。**
命令行只使用真实 Ollama，不会在连接失败时切换成模拟模型。模拟模型仅存在于 tests/。

论文：https://arxiv.org/abs/2308.10144  
作者代码：https://github.com/LeapLabTHU/ExpeL  
参考代码固定提交：`e41ec9a24823e7b560c561ab191441b56d9bcefc`

## 先运行这三步

前提：Python 3.10 或更高版本；安装 Windows 版 Ollama：https://ollama.com/download/windows 。
打开 Ollama 应用，使其后台服务保持运行。在解压后的项目目录打开 PowerShell：

```powershell
ollama pull qwen3.5:9b
python -m expel doctor
python -m expel one
```

`doctor` 实际调用模型并保存 `runs/pilot/connectivity.json`。
`one` 用经验集中的一道真实 HotpotQA 题目运行多步搜索，保存 `single_example.json`。
看其中的 `prediction`、`metrics.em`、`steps`，不要把有输出等同于答案正确。

Ollama 当前官方 Qwen3.5 小模型标签是 4B、9B 等，并没有 `qwen3.5:8b`；本项目用 9B 满足“8B 及以上”。
9B 文件下载约 6.6GB，实际运行内存/显存还需要容纳上下文等，不能把下载大小当作显存需求。
尚未拿到你的实际硬件配置，所以不能承诺运行速度或是否完全装入显存。

## 跑端到端实验

```powershell
python -m expel pipeline
```

默认：从官方仓库附带的 100 道题目中固定划分，取 8 道经验任务、4 道独立验证任务；
每道经验任务最多 1 次初始尝试 + 2 次反思重试；验证阶段每种条件每题只试一次。
默认先跑小样本，确认流程；不是论文统计规模。

默认模式除了 Python 和 Ollama，不需要安装任何 Python 第三方包，不需要 API Key。
如果想分阶段运行，下列命令等价于 pipeline：

```powershell
python -m expel collect
python -m expel extract
python -m expel evaluate
```

中途关闭后，用完全相同的命令重跑即可恢复。收集按题目保存，提炼按批次保存，验证按题目/条件保存。
正在执行但尚未保存的单个题目/批次会重新执行。改变模型、代码、数据或实验参数必须使用新的 `--out`。

## 四组对照

| 标识 | 经验规则 | 按问题相似度检索成功案例 |
|---|---|---|
| react | 不使用 | 不使用；固定示例 |
| insights_only | 使用 | 不使用；固定示例 |
| retrieval_only | 不使用 | 使用 |
| expel | 使用 | 使用 |

固定示例与检索示例数量要求相同（`--demos` 必须等于 `--top-k`）。
检索池包含经验集成功轨迹与固定论文示例；不包含验证任务轨迹。
模型参数全程不更新。所有模型调用只走 `/api/chat`，不训练、不加载 LoRA。
同一个 Qwen 模型同时承担解题、反思与规则提炼；运行前后检查模型 digest，记录版本和采样设置。

## 输出文件

| 文件（均在 runs/pilot 下） | 用途 |
|---|---|
| manifest.json | 模型摘要、Ollama 版本、配置、代码/数据指纹、分组题目 ID |
| experience.json | 成功/失败轨迹、反思记录、实际模型请求与 token 消耗 |
| insights.json | 当前规则、票数、每次规则更新的来源和原始输出 |
| evaluation.json | 各条件逐题答案、EM/F1、动作、耗时、完整请求 |
| summary.csv / summary.json | 四组统计、准备阶段开销、改进与退步题数 |
| report.md | 可直接阅读的中文实验报告 |

报告只在四组任务全部完成后产生，不把未完成任务算成已完成实验。
规则为空时停止验证并说明原因，不伪造经验；可扩大经验集后重试。
如果没有失败后成功的任务对，报告会提示尚未实际覆盖“成功/失败配对提炼”。

## 环境与复现范围

默认 `--environment local` 把 100 道题自带的百科段落合并成一个固定检索库。
这是**有限语料库上的机制验证**，不是论文的在线 Wikipedia 全站搜索，也不是标准 HotpotQA 十段落直接阅读协议。
所有条件使用同一个检索库；语料库只含公开段落，不含题目答案字段、支持事实标签。
经验规则和案例池仅由经验任务构建；验证标签只在任务结束后评分。

可选在线 Wikipedia：

```powershell
python -m expel pipeline --environment wikipedia --out runs/wiki-pilot
```

需要能够访问英文维基百科 API。采用精确标题搜索、候选标题建议和关键词查找。
请求结果缓存到同一实验目录，缓存不会让首次请求免联网；各实验目录各有缓存。
这仍是 MediaWiki 适配器，不与旧 LangChain Docstore 的每个细节完全等价。

## 更接近论文的语义检索

默认用无需依赖的 BM25 词项检索，这是工程适配。安装以下可选依赖后可用论文同名嵌入模型：

```powershell
python -m pip install -r requirements-semantic.txt
python -m expel pipeline --retriever mpnet --out runs/mpnet-pilot
```

`mpnet` 使用 `sentence-transformers/all-mpnet-base-v2`，首次需要下载权重，在 CPU 上编码问题，
使用归一化向量点积排序；没有采用 Faiss 库，小规模案例池直接精确排序。
建议先用默认模式验证连通性，再切换，避免同时排查模型下载和实验代码。

## 扩大实验

```powershell
python -m expel pipeline --train-limit 30 --eval-limit 20 --out runs/local-30-20
```

`--train-limit 0 --eval-limit 0` 表示使用当前折的全部数据，约 75 道经验题、25 道验证题。
`--fold 0` 至 `--fold 3` 分别选择四折；每一折必须重新收集经验、提炼规则，不能共用记忆。
已有 `tools/run_folds.py` 可串行运行四折并输出均值和标准误，见研究说明。
不要仅依据 4 道验证题得出有效性结论，也不保证记忆一定提升小模型表现。

## 常见错误

- `Cannot reach Ollama`：启动 Windows Ollama 应用；默认地址是 `http://127.0.0.1:11434`。
  不需要把手机/其它电脑上的 localhost 填进来；localhost 始终指运行 Python 的那台电脑。
- `Model ... missing`：先运行 `ollama list` 看真实名称，再下载 `qwen3.5:9b`。
- `unknown model` 或不识别 `think`：更新 Ollama；本项目对 Qwen3.5 设置 `think=false`，
  使用简短可见行动计划，避免把推理输出格式与动作解析混在一起。
- 内存不足/太慢：先用 `--num-ctx 8192 --out runs/ctx8k`，仍需检查机器内存和显卡。
  可用 4B 调试连通性，但要新建实验目录，4B 结果不能写成“8B 及以上”任务完成。
- 超时：增加 `--timeout 600` 并换新输出目录。默认单次 HTTP 超时 300 秒。
- `Prompt exceeds ...`：增加上下文，或减少示例/提炼批次；程序不会主动静默删除历史。
  字符预检不是精确 tokenizer，结果同时记录实际 token 数和上下文接近上限告警。
- `No learned rules`：当前经验不足、没有成功任务或模型未生成合法规则。查看 experience/insights，
  扩大经验集再运行；不要手填“学到的规则”冒充自动提炼结果。
- 实验参数变更报错：用新 `--out`，防止把不同模型/配置结果混在一起。

## 阅读代码

`agent.py`：搜索、读取反馈、继续行动的循环。  
`pipeline.py`：经验收集、反思重试、独立验证、结果汇总。  
`memory.py`：成功/失败配对、成功轨迹批次提炼、规则票数更新。  
`retrieval.py`：成功轨迹检索。  
`client.py`：真实 Ollama 调用与 token/耗时记录。  
`environment.py`：固定语料库和在线百科两种环境。

```powershell
python -m unittest discover -s tests -v
```

离线测试仅验证程序逻辑，不代表 Qwen 基准准确率。详细机制对照见 `docs/研究说明.md`。
