# WebShop 运行说明

## 已有 WebShop 服务

如果组内已有可访问的购物环境，直接使用同一个服务地址，无需再下载商品库。在 Windows 项目根目录运行：

```powershell
python -m expel webshop-check --webshop-url http://127.0.0.1:3000
python -m expel doctor --environment webshop --webshop-url http://127.0.0.1:3000
python -m expel one --environment webshop --webshop-url http://127.0.0.1:3000 --out runs/webshop-one
python -m expel pipeline --environment webshop --webshop-url http://127.0.0.1:3000 --out runs/webshop-pilot
```

服务器在其他机器上时，把地址替换为该机器的实际地址；`127.0.0.1` 指运行 Python 的这台电脑。

## 从零安装服务：WSL + 官方代码

WebShop 官方安装涉及 Bash、Java、商品文件和搜索索引。本方案让购物服务运行在 WSL Ubuntu，模型和实验程序仍运行在 Windows。以下是基于官方脚本的安装步骤，尚未在真实 Windows/WSL 上验收。

1. 在管理员 PowerShell 安装 WSL；若系统要求重启，重启后继续：

```powershell
wsl --install -d Ubuntu
```

2. 在 Ubuntu 中安装 **Linux 版 Miniconda**，按其安装说明启用 `conda`；已安装则跳过。官方安装说明：https://www.anaconda.com/docs/getting-started/miniconda/install 。不要使用 Windows 的 conda 环境代替 Ubuntu 内的环境。

3. 下列命令都在 **Ubuntu 终端** 中运行：

```bash
conda create -n webshop python=3.8.13 -y
conda activate webshop
cd ~
git clone https://github.com/princeton-nlp/WebShop.git webshop
cd ~/webshop
git rev-parse HEAD > installed_revision.txt
bash setup.sh -d all
```

官方脚本安装依赖、通过 Google Drive 下载商品与指令、下载 spaCy 模型并创建搜索索引；需要网络和较长的安装时间。脚本中 conda 可能提示确认安装。下载失败或索引构建报错应先解决，不能把脚本执行到末尾当成安装成功。

官方旧版 Flask 若出现 `cannot import name url_quote from werkzeug.urls`，在 **webshop 环境**中运行下面命令后重新启动：

```bash
python -m pip install Werkzeug==2.1.2
```

4. 官方服务默认可能仍选择 1000 商品文件。正式使用作者任务子集时，建议与组内实验一致使用完整商品库；下载完整数据后，在 `~/webshop/web_agent_site/utils.py` 中把两项路径改为：

```python
DEFAULT_ATTR_PATH = join(BASE_DIR, '../data/items_ins_v2.json')
DEFAULT_FILE_PATH = join(BASE_DIR, '../data/items_shuffle.json')
```

若存在 `DEBUG_PROD_SIZE` 限制，设置为 `None`。也可使用本项目附带脚本自动修改并保留原文件备份。在 Ubuntu 中执行，替换引号内路径为实际解压位置（Windows 的 D 盘对应 `/mnt/d/`）：

```bash
python "/mnt/d/Users/lenovo/Downloads/ExpeL-WebShop-Ollama/expel-local/tools/configure_webshop.py" "$HOME/webshop"
```

5. 在 Ubuntu 启动购物服务器，并保持该终端运行：

```bash
conda activate webshop
cd ~/webshop
bash run_dev.sh
```

按官方默认配置使用 3000 端口；首次加载商品和目标可能较慢。待服务就绪后，在 Windows 浏览器打开 `http://127.0.0.1:3000/fixed_0` 应看到购物指令。仅在自己电脑内运行不需要暴露端口到公网。

## Windows 上执行实验

启动 Windows Ollama，确认 `ollama list` 中有 `qwen3.5:9b`。没有时先下载：

```powershell
ollama pull qwen3.5:9b
```

在解压后的 `expel-local` 目录执行：

```powershell
python -m expel webshop-check
python -m expel doctor --environment webshop
python -m expel one --environment webshop --out runs/webshop-one
python -m expel pipeline --environment webshop --out runs/webshop-pilot
```

第一次只跑 8 条经验、4 条验证，确认模型能完成真实搜索和购买。之后运行：

```powershell
python -m expel pipeline --environment webshop --train-limit 30 --eval-limit 20 --webshop-catalog full --out runs/webshop-30-20
```

只有确实使用完整商品库时才填写 `--webshop-catalog full`。

若 `task mismatch`：服务器的固定编号与包内公开任务不一致，导出当前服务的指令，并在后续命令中明确使用：

```powershell
python -m expel webshop-export --task-count 100 --export-path data/webshop_server_tasks.json
python -m expel doctor --environment webshop --data data/webshop_server_tasks.json
python -m expel one --environment webshop --data data/webshop_server_tasks.json --out runs/webshop-server-one
python -m expel pipeline --environment webshop --data data/webshop_server_tasks.json --out runs/webshop-server-pilot
```

改变配置时使用新的输出目录。连接超时可加 `--webshop-timeout 600`；模型调用超时则加 `--timeout 600`。商品环境报错与 Ollama 报错是两个独立问题，先通过 `webshop-check` 再检查模型。

## 结果怎么看

`single_example.json` 的 `steps` 应包含服务器返回的商品页面，最终购买后有真实 `reward`。完整运行的 `report.md` 给出四组成功率、平均奖励、步数、耗时、经验构建成本及结果发生变化的任务数。

成功率是奖励为 1.0 的比例；部分满足购物条件仍可能有非零奖励，但不计完整成功。当前包没有真实 WebShop 实验成绩；旧 HotpotQA 成绩不能转写为 WebShop 成绩。

参考：WebShop 官方安装和运行说明 https://github.com/princeton-nlp/WebShop ，作者 ExpeL 环境配置 https://github.com/LeapLabTHU/ExpeL 。
