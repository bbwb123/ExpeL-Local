# 交付验证记录 — WebShop 适配

- 原有 13 项离线测试继续通过；新增 10 项 WebShop 测试，共 23 项通过。
- 新增测试覆盖任务转换与两折隔离、HTML 观察、规格选择、分页与返回、独立购物会话、指令匹配、真实 HTTP 请求序列、官方奖励字段读取、部分奖励、步数耗尽、无效输出隔离及完整流程断点恢复。
- 测试服务提供的是合成页面，模型为脚本模型，只验证程序逻辑，不属于实际 WebShop 数据集评测。
- 包内 100 条公开任务和 2 条演示来自 ExpeL 作者固定提交；目标商品字段已从运行任务中移除。
- 制作环境未运行 Ollama，也未安装带完整商品数据和索引的官方 WebShop 服务，因此尚无真实 Qwen/WebShop 成绩，Windows/WSL 安装步骤亦待用户电脑验收。
- 原有 HotpotQA 实验结果属于另一环境，不能当作 WebShop 实验结果。本包不预填 `runs/` 数据。

真实验收顺序：`webshop-check` → `doctor --environment webshop` → `one --environment webshop` → `pipeline --environment webshop`。
