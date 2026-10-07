# WebShop 复现：agent 不点商品、reward 恒为 0 的排查与修改记录

环境：Windows + Ollama `qwen3.5:9b`（Q4_K_M）；WebShop 官方 Flask 服务（WSL，`http://127.0.0.1:3000`，small 商品库）；
任务文件 `data/webshop_small_synth.json`（自动生成指令）；max_steps=15，2 个固定示范。

## 1. 现象与诊断

原始现象：agent 只在结果页反复 `search`，收到 `Invalid action: click[Back to Search] before a new search.` 仍照样重复，
从不 `Buy Now`，reward=0。逐步查看轨迹后定位到三个原因：

1. **think=false 时 Thought 是“先下结论后看结果”**：第 2 步几乎固定输出“results do not match … refine the query”，
   然后在结果页直接 `search`（非法）。temperature=0 下上下文一重复，输出就逐字重复，形成
   `search → 非法 search → Back to Search → 同一个 search` 的死循环。
2. **过度挑剔，从不购买**：原提示词写的是 “Click Buy Now **only when** the product, options and price meet the instruction”。
   模型打开商品后只要有一个属性对不上（例如没有 `157- green` 颜色）就 `Back to Search` 重新搜同样的词，
   得到同样的结果、点开同一个商品，循环到步数耗尽。实际上 WebShop 的 reward 是部分得分，买“最接近的”远好于不买。
3. **不知道步数预算**：模型不知道只剩几步，最后一步还在浏览，导致即使看到了合适商品也没买。

网页解析、动作合法性检查和 reward 读取都没有问题（例如商品页的 color/size 选项能正确解析成 `[a2-gray]` 等可点击标签）。

## 2. 实验（8 个经验集任务 fixed_23/11/45/9/16/21/3/38，单 seed）

用 `probe.py` 跑 baseline actor（与 ExpeL 共用的 `run_webshop_episode`），结果保存在 `runs/probe/`：

| 编号 | 提示词 | think | temperature | 平均 reward | 成功 | 完成购买 | 非法动作/总步数 | 秒/任务 |
|---|---|---|---|---|---|---|---|---|
| A | 原（含 3 行导航提示） | 0 | 0   | 0.250 | 2/8 | 2/8 | 27/113 | 50 |
| B | 原 | 0 | 0.5 | 0.250 | 2/8 | 2/8 | 17/104 | 40 |
| C | 新提示词 | 0 | 0 | 0.250 | 2/8 | 2/8 | 22/103 | 39 |
| D | 原 | 1 (num_predict 4096) | 0 | 0.375 | 3/8 | 3/8 | 2/97 | 122 |
| E | 新提示词 | 1 | 0 | 0.625 | 5/8 | 5/8 | 6/88 | 121 |
| **F** | **新提示词 + 步数预算** | **1** | **0** | **0.643** | **5/8** | **7/8** | 4/85 | 126 |
| G | 新提示词 + 步数预算 | 0 | 0 | 0.500 | 4/8 | 4/8 | 16/85 | 42 |

结论：
- **temperature=0.5 没有作用**（A vs B）：只是把确定性循环变成随机循环，模式不变。
- **think=1 是让模型“读完结果再决定”的关键**：非法动作从 ~24% 降到 ~2%，开始真正点进商品；代价是每任务耗时约 2.5–3 倍。
  注意 think=1 时思考 token 也计入 `num_predict`，需要 `--num-predict 4096`（1024 会截断）。
- **提示词修改只有在 think=1 时才明显生效**（D→E：0.375→0.625）；think=0 时单改提示词无效（A→C）。
- **步数预算**主要把“看到了但没买”变成“买了”（E→F：购买 5/8→7/8），think=0 下也有提升（C→G：0.25→0.50）。
- 推荐配置：**F（新提示词 + 步数预算 + think=1 + num_predict 4096 + temperature 0）**。
  若时间预算紧，G（think=0）是次优选择。
- 局限：n=8、单 seed，差异 1 个任务 = 0.125；think=1 下 GPU 推理并非完全确定（同一任务在 `one` 与 probe 中结果可能不同）。
  正式汇报请用 validation 集和多个 fold/seed。

用 CLI 复核（与 F 相同配置）：`runs/ws_one_final/`、`one_final.json`，fixed_23 reward=1.0（点商品 → 选 a2-gray / large → Buy Now）。

## 3. 代码改动（全部同时作用于 baseline 与 ExpeL）

ExpeL 的四个条件（react / insights_only / retrieval_only / expel）在 collect 和 evaluate 阶段都调用同一个
`expel/webshop.py: run_webshop_episode`，客户端也是同一个 `Ollama` 实例，因此下面的改动对 baseline 和 ExpeL 一致。
存入 trajectory（供反思、规则抽取、检索示范使用）的仍是原始 observation，不含 Available actions 与步数提示。

### 3.1 之前已做（用户）
| 文件 | 改动 | 备份 |
|---|---|---|
| `expel/webshop.py` | INSTRUCTIONS 末尾加 3 行导航提示 | `webshop.py.bak` |
| `expel/webshop.py` | 新增 `available_actions(env)`，每步 observation 后附 `Available actions: ...` | `webshop.py.bak2` |
| `expel/client.py` | `think`、`temperature` 改为读环境变量 `EXPEL_THINK`、`EXPEL_TEMPERATURE`（默认 0/0，与原行为一致） | `client.py.bak` |
| `show.py` | 查看单次运行轨迹 | — |

### 3.2 本次修改
| 文件 | 改动 | 备份 |
|---|---|---|
| `expel/webshop.py` | **INSTRUCTIONS 重写后半段**：删除 “Buy Now only when … meet the instruction”；说明 Buy Now 是**部分得分**、不买得 0、必须以 Buy Now 结束；导航改为“搜索一次 → 点最相关商品（标题不必包含全部属性）→ 选匹配选项 → Buy Now；不合适就 `< Prev` 回到**同一结果页**看下一个；不要为同类查询点 Back to Search；结果页上 search 必然非法”；以及“剩余步数不多时回到见过的最佳商品下单”。 | `webshop.py.bak3`（改提示词前） |
| `expel/webshop.py` | **步数预算**：`run_webshop_episode` 在初始观察和每步观察末尾附 `(Steps used: i of 15; N left. Buy before they run out.)` | `webshop.py.bak4`（加步数前） |
| `expel/client.py` | `EXPEL_THINK` / `EXPEL_TEMPERATURE` 改为在 `Ollama.__init__` 读一次存为 `self.think` / `self.temperature`（行为不变） | `client.py.bak2` |
| `expel/pipeline.py` | `manifest.json` 新增 `sampling: {think, temperature, num_predict}`，保证不同采样设置的运行能被区分、写进报告；它参与 `experiment_fingerprint`，所以旧运行目录不能用新代码续跑（请用新 `--out`） | `pipeline.py.bak` |
| `probe.py`（新增） | 在 8 个经验集任务上跑 baseline actor 并打印每任务一行摘要；支持 `EXPEL_PROMPT_FROM=<备份文件>` 做提示词 A/B | — |

### 3.3 第一次完整 pipeline 后的修正（2026-10-06）
| 文件 | 改动 | 备份 |
|---|---|---|
| `expel/client.py` | `think` 只作用于 actor 调用（`purpose == 'agent'`，baseline 与 ExpeL 四个条件相同）；反思与规则抽取恢复原项目的 think=false。调用记录新增 `think` 字段；空输出报错信息带上 purpose/think/done_reason | `client.py.bak3` |
| `expel/pipeline.py` | manifest `sampling` 增加 `think_scope: "agent"` | `pipeline.py.bak2` |

原因：`runs/ws_expel_server` 的 extract 第 2 组（contrast 组，prompt 约 4.4k token）在 think=1 下把 4096 个
`num_predict` 全部用于思考（done_reason=length，思考约 1.75 万字符），content 为空 → `Ollama returned empty content`。
修正后同一组 evidence 用 think=false 重试，两组均在约 50 个 token 内返回合法 JSON（`applied`）。
这样也更贴近原 ExpeL：变量只有 actor 是否思考，记忆构建方式与原项目一致。

**WebShop 任务价格不一致**：官方 `web_agent_site/engine/goal.py` 生成 synthetic goal 时用 `random.sample` 抽
`price_upper`，而 `app.py` 的 `random.seed(233)` 在生成之后才设置，所以**每次重启 WebShop 服务，任务顺序不变但价格上限会变**
（例如 fixed_23 从 30.00 变成 40.00）。因此：
- 每次重启服务后都要重新 `webshop-export` 并用新的任务文件；一次实验（collect→extract→evaluate）期间不要重启服务；
- 不同服务进程下的运行（包括第 2 节的 `webshop_small_synth.json` 探针）的任务并不完全相同，只能做定性对比。

**上下文预算（`--num-ctx`）**：`runs/ws_expel_server_v2` 在 collect 4/8 的重试（attempt=1）时触发
`Prompt exceeds conservative context budget`。这是 `client.py` 的保守预检（总字符数 > 2 × num_ctx = 32768），
触发时 prompt 确实超过 32768 字符，但未必超过 16384 token：v2 中由 Ollama 报告的 prompt token 最大为 9.2k，
prompt+输出最大 11.4k。（更正：此前写的“32k 字符 ≈ 7.4k–9.2k token、3.5–4.3 字符/token”不成立，
那是用被后续对话改写的调用日志算出来的，见 3.4；字符/token 比例无法从 v2/v3 日志中准确计算。）
但长度来源是结构性的：每步观察平均约 780 字符，加上 Available actions、步数提示和模型输出，每步约 1.5k 字符；
15 步的 baseline episode 就接近 30k 字符，重试时再加约 5k 字符的反思；ExpeL 条件还要放入 2 条检索到的成功轨迹
（当前每条 6–8k 字符，最长可到约 20k）。所以 16384 对 ExpeL 条件本来就不够。
**处理：所有条件统一用 `--num-ctx 32768`**（预检上限 65k 字符；按英文常见的约 3.5–4 字符/token 估算约 16–19k token，加 4096 输出仍在 32768 以内），不改代码。
代价（RTX 4060 Laptop 8GB 实测）：显存占用基本不变（多出的部分放在 CPU），生成速度从 39 降到 31 token/s（约 -22%）。
num_ctx 只决定能放下多长的上下文，放得下的 prompt 不会被改动，对 baseline 和 ExpeL 一视同仁。

### 3.4 审阅意见修正（2026-10-06）
| 文件 | 改动 |
|---|---|
| `expel/client.py` | 调用记录保存 `copy.deepcopy(messages)`。此前 `record['messages']` 引用的是同一个列表，之后追加观察/反思时，早期调用的日志也会被改写。已复现：v3 中前 4 次 actor 调用的日志都显示 30 条消息，而 Ollama 报告的 prompt token 从 2.2k 逐步增至 3.6k。**只影响日志审计，不影响模型实际看到的内容**（请求在记录之前已发出）。v1–v3 的 `calls[].messages` 因此不能用来逐步还原 prompt；`steps`、`prompt_tokens`、reward 不受影响。 |
| `expel/webshop.py` | 格式错误被拒绝时同样消耗一步，拒绝提示现在也附上 Available actions 和剩余步数（原来只有正常观察才附）。 |
| `tests/test_core.py` | 新增日志快照测试、think 只作用于 actor 的测试；`test_native_ollama_payload_and_usage` 固定 `EXPEL_THINK=0`，不再受 shell 环境变量影响（原来在 `EXPEL_THINK=1` 下运行测试会失败）。 |

未改：15 步上限、reward=1.0 才算成功、方法本身。
核对：`data/webshop_server_tasks.json` 的 100 条指令与当前 WebShop 服务逐条一致（服务导出后未重启）。

单元测试：25 个全部通过（`PYTHONUTF8=1`，`EXPEL_THINK=1` 与 `0` 两种设置下都已运行；不加 `PYTHONUTF8=1` 时
`test_webshop` 有 1 个测试因 GBK 区域设置读 report.md 报 UnicodeDecodeError，与本次改动无关）。

未修改：WebShop 环境语义（动作合法性、页面状态机、reward 读取）、示范、ExpeL 的反思/规则抽取/检索逻辑。

## 4. 复现命令（PowerShell）

```powershell
$env:EXPEL_THINK = '1'; $env:EXPEL_TEMPERATURE = '0'
# 单任务
python -m expel one --environment webshop --data data\webshop_small_synth.json --webshop-catalog small --num-predict 4096 --out runs\ws_one_final > one_final.json
python show.py one_final.json
# 8 任务探针（EXPEL_NUM_PREDICT 只对 probe.py 生效）
$env:EXPEL_NUM_PREDICT = '4096'; python probe.py runs\probe\F_budget_think.json 8
# 完整 baseline vs ExpeL
python -m expel pipeline --environment webshop --data data\webshop_server_tasks.json --webshop-catalog small --num-predict 4096 --num-ctx 32768 --train-limit 10 --eval-limit 10 --out runs\ws_review_10_10
```

注意：`one` 的结果同时保存在 `runs\<out>\single_example.json`，若重定向得到的文件编码有问题，可直接 `python show.py runs\<out>\single_example.json`。

## 5. 第一次完整运行结果：`runs/ws_expel_server_v3`（2026-10-06）

配置：think=1（只作用于 actor）、temperature=0、num_predict=4096、num_ctx=32768，经验集 8 题、验证集 4 题，max_steps=15。

| 条件 | 成功率 | 平均奖励 | 平均秒数 |
|---|---:|---:|---:|
| react | 0.250 | 0.250 | 196 |
| insights_only | 0.250 | 0.250 | 170 |
| retrieval_only | 0.000 | 0.214 | 224 |
| expel | 0.250 | 0.350 | 221 |

解读（写汇报时请注明）：
- **样本太小，不能下结论**：每个条件只有 4 题。ExpeL 比 react 多出的 0.10 全部来自 fixed_65 的一次部分得分（0.4），
  4 题里有 3 题四个条件都没有**完整成功**（更正：此前写成“都没买到”不准确，fixed_65 的 expel 买了，得 0.4）；
  逐题对照中“baseline 错、ExpeL 对”的成功题数为 0。
- **ExpeL 的核心机制没有被触发**：collect 阶段第一次就成功的题（4/8）没有失败轨迹，反思重试的题（4/8）一题也没有重试成功
  （fixed_41 第一次得 0.9，反思后两次都是 0），所以成功/失败对比组 = 0，只用 2 个成功批次提炼出 1 条规则
  （“点 Description/Features 后先 `< Prev` 回商品页再 Buy Now”），属于操作层面的规则，对选商品帮助不大。
- **主要失败方式仍是步数耗尽**：验证集 16 个 episode 中 11 个是 max_steps，典型模式是 `S C C S C C …`
  （搜索 → 看 1–2 个商品 → 重新搜索）。
- `evaluation.json` / `show.py` 中 max_steps 的 episode 也有 `prediction`（ASIN），那只是**最后打开的商品**，不代表买了；
  是否购买看 `status == "finished"`。
- 成本：collect+extract 共 242 次调用、约 78 分钟；evaluate 16 个 episode 约 54 分钟。

## 6. 任务来源核查（2026-10-06，回应审阅）

**任务如何生成**：WSL `~/webshop` 相对官方仓库（HEAD 64fa2a5）的实际改动只有 `web_agent_site/app.py` 两行：
`load_products(..., human_goals=False)` 与 `get_goals(..., human_goals=False)`，即改用 synthetic goals
（diff 见 `review_bundle/webshop_app_local_changes.diff`；其余 git 显示的修改均为换行符差异）。
`engine/goal.py` 未修改（WSL 与 Windows 两份 SHA-256 均为 `2ca0fa60…8b90`）。`get_synthetic_goals` 不现场生成文字：
指令主体和 `instruction_attributes` 直接取自官方预生成文件 `data/items_ins_v2_1000.json`，服务启动时只追加
选项组合（`with color: …, and size: …`）和随机价格上限。

**`dry clean` 冲突来自官方数据，不是本项目引入**：fixed_42 的目标商品 B01HQTWL6S 在 `items_ins_v2_1000.json` 中的
`instruction_attributes` 为 `['machine wash', 'wash cold', 'dry clean', 'tumble dry']`，而商品原文两处写
“Do not iron. Do not dry clean.”——属性抽取丢掉了否定词。用简单正则（“do not/don't/not/no + 属性”）扫描全部 415 个
有指令属性的商品，命中 13 个候选（正则很粗，除 B01HQTWL6S 外未逐个人工确认）；落在本实验 100 个任务里的只有
B01HQTWL6S，对应 **fixed_17、fixed_42、fixed_45** 三题。处理原则：不修改任务文本、不删属性、不改评分；汇报中把这三题
标注为“指令与商品描述存在冲突（官方数据问题）”，并在分析中单独列出。

交给审阅的文件（`review_bundle/`，不入 git）：`experience_v3.json`（= `runs/ws_expel_server_v3/experience.json`，
SHA-256 `cf65ea73…d984`）、`goal.py`（WSL 原文件）、`webshop_app_local_changes.diff`。
注意：v1–v3 的 `calls[].messages` 有 3.4 所述的日志改写问题，审阅反思/重试时请以 `steps` 和 `reflections_used` 为准。

## 7. 反思提示词调整（2026-10-06，审阅建议，工程调整）

**改动**：`expel/webshop.py` 的 `reflect_webshop` 换成通用约束版（备份即 git 提交 `2e17248` 中的旧版）：
≤120 词；区分观察事实与推测；不要只凭总分推断哪个属性丢分；先检查漏选选项、重复搜索、非法动作和浪费的步数；
提示“重新打开商品可能清空已选选项”（本适配器中点击商品 ID 或从商品页 `< Prev` 都会清空 options）；只能提出
search/click/think 及页面上可见的控件，不得编造筛选、滚动或“提交答案”动作；在步数预算内计划，给选项和 Buy Now
留步数，避免无止境地找完美匹配；不得编造商品事实或建议改动购物指令。user 消息新增 Status、Reward、
Final selected options。不含任何题目的答案或评分细节。

**依据（v3 的 10 条反思）**：
- 长度 270–590 词，远超可用的重试计划；
- 凭总分推断丢分属性并要求更严格的匹配：fixed_41 首次 0.9，反思认定 “polyester cotton/moisture wicking”
  未满足，此后两次都耗尽步数得 0；fixed_56 三次都是 0.714，反思归因于颜色代码 `228l.brown` 未精确匹配（未经核实）；
- 建议不存在的操作：“failed to filter for the category”、外部品牌名（如 Victoria's Secret、L'Occitane）；
- fixed_45 的反思认为 “dry clean 与 tumble dry 互斥”、选了 “Do not dry clean” 的商品是错误——与评分器不一致（见下）。

**评分器核对（`goal.py: get_attribute_reward`）**：先用 `fuzz.token_set_ratio` 与商品自身的属性列表（B01HQTWL6S
的列表本身含 `dry clean`）匹配，否则检查属性字符串是否出现在标题/要点/描述中，均不处理否定。因此 fixed_17/42/45
在评分器下是可解的，模型按自然语言判断“不能干洗→不符合”会与评分不一致。这是对第 6 节的补充：数据中丢失否定
（items_ins_v2_1000.json）与评分器不处理否定同时存在；处理原则不变——不改任务、不改评分、**也不把这一评分细节
写进提示词**。

影响范围：反思只在 collect 阶段的重试中使用，进而影响经验与规则；四组评测共用同一份修改后的代码。
保持 15 步、reward=1.0 才算成功。v3 保留作记录，新实验用新目录 `runs/ws_reflection_v4`（10 经验 + 10 验证）。
单元测试 25 个通过；`data/webshop_server_tasks.json` 与当前服务 100 条一致。

## 8. 生成超限不再中断实验（2026-10-06）

`runs/ws_reflection_v4` 在评测第 3/10 题的 expel 条件中断：actor（think=1）一次回复用满 4096 个 `num_predict`
仍无可见内容（done_reason=length），抛出 `Ollama returned empty content`，整个 pipeline 退出。
这是偶发的失控生成，而非常态：v4 collect 的 217 次 actor 调用最长 1216 token，已完成的 11 个评测 episode 最长 826 token。
所以单纯调大 `--num-predict` 不能保证不再发生。

**改动**：`expel/client.py` 新增 `TruncatedOutput`（done_reason=length 且无内容时抛出，其他空输出仍是 RuntimeError）；
`run_webshop_episode` 捕获它并记为一次被拒绝的步骤（与格式错误相同：消耗 1 步，提示
“Generation limit reached before any Action. Think briefly, then answer.”，并附可用动作和剩余步数）。
调用日志照常记录（`done_reason: length`），可统计各组发生次数。四组共用；反思/规则抽取（think=false）行为不变。
新增 2 个测试，共 27 个通过。

代码变化使实验指纹改变，v4 不能续跑（保留作记录）。新实验目录 `runs/ws_reflection_v5`，配置与 v4 相同。

## 9. 10/10 运行结果：`runs/ws_reflection_v5`（2026-10-07）

配置同 v4（think=1 仅 actor、temperature=0、num_predict=4096、num_ctx=32768、15 步、reward=1.0 才算成功），
代码为第 7 节反思提示词 + 第 8 节截断处理。Python 3.12.14（标准库，版本见 manifest runtime）。

collect：10 题成功 7 题（5 题首次成功；重试后成功 2 题：第 1 题第 2 次、第 8 题第 3 次；3 题三次均未成功），
每次尝试的 reward 与中断前的 v4 逐一相同。extract：3 个成功/失败对比组 + 4 个成功批次，最终 3 条规则。

| 条件 | 成功率 | 平均奖励 | 未购买（跑满 15 步） | 环境拒绝 | 格式错误 | 生成截断 | 平均秒数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| react | 3/10 | 0.383 | 6 | 9 | 0 | 0 | 171 |
| insights_only | 4/10 | 0.483 | 5 | 10 | 0 | 0 | 162 |
| retrieval_only | 1/10 | 0.298 | 6 | 9 | 0 | 0 | 210 |
| expel | 3/10 | 0.413 | 5 | 10 | 0 | 1 | 213 |

逐题 reward（* = 未购买）：

| 任务 | react | insights_only | retrieval_only | expel |
|---|---:|---:|---:|---:|
| fixed_42 | 0* | 0* | 0* | 0* |
| fixed_91 | 0* | 0* | 0* | 0* |
| fixed_65 | 0* | 0* | 0* | 0* |
| fixed_1 | 1 | 1 | 0.857 | 1 |
| fixed_15 | 0* | 0* | 0* | 0.300 |
| fixed_73 | 0.833 | 0.833 | 0.833 | 0.833 |
| fixed_55 | 1 | 1 | 1 | 1 |
| fixed_72 | 1 | 1 | 0* | 1 |
| fixed_48 | 0* | 1 | 0.286 | 0* |
| fixed_76 | 0* | 0* | 0* | 0* |

观察（仅描述轨迹，n=10 不足以下统计结论）：
- 与 baseline 相比，成功题的差异只有 1 题：fixed_48 仅 insights_only 成功。expel 与 react 的成功题完全相同
  （报告中“baseline 错/ExpeL 对”与反向均为 0）；expel 平均奖励更高只来自 fixed_15 的 0.3 部分得分。
- fixed_42（dry clean 冲突）：四组都没买。expel 第 6 步已打开商品并读到描述，判断“requires for dry clean. This is a mismatch”
  后返回，此后耗尽步数——与审阅在 v3 中看到的现象一致。其余三组的模型输出中没有提到 dry clean，它们是在搜索循环里耗尽步数。
- fixed_1 retrieval_only 再次只选了 size、漏选 color: black（0.857），与 v3 相同。
- fixed_72 retrieval_only 在商品页反复出现环境拒绝（最后 3 步都是非法动作）而没买，其余三组都满分。
- 主要失败方式仍是搜索循环：40 个 episode 中 22 个未购买；fixed_91、fixed_76 四组都是 `S C C S …` 循环。
- 生成截断只出现 1 次（fixed_65 expel 第 15 步，与 v4 中断位置相同），已按第 8 节记为一步，未中断实验。

审阅材料：`review_bundle/ws_reflection_v5/`（evaluation/experience/insights/manifest.json 与 report.md）。
本轮日志已是深拷贝，`calls[].messages` 可以逐步还原 prompt。
