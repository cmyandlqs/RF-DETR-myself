# RF-DETR 本地开发与训练协作说明

## 目标

本项目当前只服务于以下主线：

1. 在当前 `RF-DETR` 代码上跑通自己的 `VisDrone` 数据集。
2. 在可运行基线之上持续改进，逐步形成论文工作。
3. 采用 `Windows 本地开发 + 服务器训练测试` 的协作方式。
4. 数据准备阶段优先把 `VisDrone` 转成当前仓库可直接消费的格式。

## 当前约定

- 数据集目录：`visdrone/`
- 本地环境：Windows
- 代码开发与调试：本地完成
- 正式训练与测试：代码同步到服务器后执行
- 训练过程记录：统一使用 `WanLab`
- 环境管理：统一使用 `conda`
- 依赖安装：在 `conda` 环境中使用 `pip` / `uv pip`

## 当前阶段需求重述

当前下一步的任务是：

1. 读懂整个项目中和训练、依赖、数据集相关的主链路。
2. 给出适合你当前协作方式的安装命令，并写进本文档。
3. 提供一个服务器可执行的环境检查脚本，打印结果给我分析。
4. 对比 `RF-DETR` 需要的数据集格式和你当前 `VisDrone` 的真实格式。
5. 如果格式不兼容，提供转换脚本。
6. 准备一套最小 smoke 流程，用来验证“环境 -> 数据 -> 训练入口”是否打通。

## 协作方式

### 本地

- 先完成数据集适配、配置修改、脚本整理、基础调试。
- 本地优先解决“能跑通”和“流程可复现”。
- 改动尽量小而清晰，便于后续上传到服务器。

### 服务器

- 通过 `git push` / `git pull` 同步代码。
- 服务器主要承担正式训练、评估、对比实验。
- 训练日志、指标和实验记录以 `WanLab` 为主。

## 开发优先级

1. 跑通 `VisDrone`
2. 固定训练与验证流程
3. 整理实验配置
4. 再开始模型或方法改进
5. 最后沉淀为论文实验体系

## 项目阅读后的关键结论

### 依赖

- 主依赖定义在 `pyproject.toml`
- 训练相关扩展在 `.[train]`
- CLI 相关扩展在 `.[cli]`
- 日志相关扩展在 `.[loggers]`

### 训练入口

- CLI 入口：`rfdetr`
- 典型入口：`rfdetr fit --config configs/rfdetr_small.yaml`
- Lightning 训练入口在 `src/rfdetr/training/cli.py`

### 数据集格式

当前仓库原生支持：

1. COCO 官方目录
2. Roboflow 风格 COCO 目录
3. YOLO 风格目录

其中对你最直接的是 Roboflow 风格 COCO 目录，要求如下：

```text
dataset_root/
  train/
    *.jpg
    _annotations.coco.json
  valid/
    *.jpg
    _annotations.coco.json
  test/
    *.jpg
    _annotations.coco.json
```

### 训练日志

当前代码中显式支持的 logger 开关是：

- `CSVLogger`
- `TensorBoardLogger`
- `WandbLogger`
- `MLFlowLogger`

也就是说，从代码层面看，当前并没有直接写死 `WanLab` logger。
如果你的 `WanLab` 方案是兼容 `wandb` 的上报方式，那么优先走 `wandb` 这条链路；
如果不是，后续需要单独补一个适配层。

## 推荐环境方案

### 本地 Windows 开发环境

优先使用 `Python 3.10`，不要直接复用你当前的 `Python 3.13` 基础环境。

当前我读到的现状是：

- 当前解释器：`Python 3.13`
- `torch` 未安装
- `pytorch_lightning` 未安装

因此建议新建独立环境：

```powershell
conda create -n rfdetr310 python=3.10 -y
conda activate rfdetr310
python -m pip install --upgrade pip
pip install uv
uv pip install -e ".[train,loggers,cli]"
```

如果你不想用 `uv pip`，可以直接：

```powershell
conda create -n rfdetr310 python=3.10 -y
conda activate rfdetr310
python -m pip install --upgrade pip
pip install -e ".[train,loggers,cli]"
```

### 服务器训练环境

服务器建议也建独立环境：

```bash
conda create -n rfdetr310 python=3.10 -y
conda activate rfdetr310
python -m pip install --upgrade pip
pip install uv
uv pip install -e ".[train,loggers,cli]"
```

如果服务器上的 PyTorch 需要和 CUDA 严格匹配，那么先根据服务器 CUDA 情况装对应 `torch/torchvision`，再执行：

```bash
uv pip install -e ".[train,loggers,cli]"
```

## 环境检查脚本

新增脚本：

- `scripts/check_env.py`

用途：

- 打印 Python、conda、git、uv 状态
- 打印 `torch` / `lightning` / `transformers` / `pycocotools` 等依赖状态
- 打印 GPU / CUDA 信息
- 打印 `visdrone/` 数据集基础统计

服务器上运行：

```bash
python scripts/check_env.py --dataset-dir visdrone --json
```

运行后把完整输出发给我，我就能继续判断服务器的安装缺口。

## VisDrone 现状与 RF-DETR 输入格式差异

### 你当前的 VisDrone

当前 `visdrone/` 真实结构是：

```text
visdrone/
  train/
    *.jpg
  test/
    *.jpg
  train.jsonl
  test.jsonl
```

并且：

- `train.jsonl` 共 `6471` 条
- `test.jsonl` 共 `548` 条
- 类别共 `10` 类：
  - `pedestrian`
  - `people`
  - `bicycle`
  - `car`
  - `van`
  - `truck`
  - `tricycle`
  - `awning-tricycle`
  - `bus`
  - `motor`

### 与 RF-DETR 的差异

差异主要有三点：

1. 当前是 `jsonl`，不是 COCO `json`
2. 当前没有 `valid/` 划分
3. 当前图片引用路径和真实落盘路径不完全一致

结论：

- 当前 `visdrone/` 不能直接喂给 RF-DETR
- 需要先转换成 Roboflow 风格 COCO 目录

## 数据转换脚本

新增脚本：

- `scripts/prepare_visdrone_for_rfdetr.py`

对应核心实现：

- `scripts/visdrone_tools.py`

默认行为：

1. 读取 `visdrone/train.jsonl` 和 `visdrone/test.jsonl`
2. 从 `train.jsonl` 按比例切出 `valid`
3. 输出为 RF-DETR 可直接读取的 Roboflow-COCO 目录
4. 默认优先使用硬链接，失败后回退复制，减少额外磁盘占用

推荐命令：

```powershell
python scripts/prepare_visdrone_for_rfdetr.py `
  --input-dir visdrone `
  --output-dir data/visdrone_rfdetr `
  --val-ratio 0.1 `
  --seed 42 `
  --image-mode auto
```

输出目录会是：

```text
data/visdrone_rfdetr/
  train/
  valid/
  test/
```

其中每个 split 里都会生成 `_annotations.coco.json`。

## Smoke 流程

新增脚本：

- `scripts/smoke_visdrone_pipeline.py`

用途：

- 使用转换后的数据集
- 构造一个 `RFDETRNano` 的最小训练配置
- 用 `fast_dev_run` 跑最小训练链路
- 验证数据集读取、dataloader、model、trainer 是否贯通

服务器建议流程：

1. 先检查环境

```bash
python scripts/check_env.py --dataset-dir visdrone --json
```

2. 再转换数据

```bash
python scripts/prepare_visdrone_for_rfdetr.py \
  --input-dir visdrone \
  --output-dir data/visdrone_rfdetr \
  --val-ratio 0.1 \
  --seed 42 \
  --image-mode auto
```

3. 最后跑 smoke

```bash
python scripts/smoke_visdrone_pipeline.py \
  --dataset-dir data/visdrone_rfdetr \
  --output-dir output/visdrone_smoke \
  --fast-dev-run 2 \
  --batch-size 1
```

## 正式训练基线

### 推荐顺序

建议先跑：

1. `RF-DETR Nano` 基线
2. `RF-DETR Small` 对比基线

原因：

- `Nano` 更稳，更快出第一版结果
- `Small` 更适合后续作为论文中的更强基线
- 先用 `Nano` 验证完整训练流程，比直接上更大的模型更省时间

### 新增配置文件

- `configs/rfdetr_visdrone_nano.yaml`
- `configs/rfdetr_visdrone_small.yaml`

两份配置都默认使用：

- `num_classes: 10`
- `dataset_file: roboflow`
- `dataset_dir: data/visdrone_rfdetr`

说明：

- 当前服务器这套 `LightningCLI + Pydantic` 组合对 `RFDETRNanoConfig` / `RFDETRSmallConfig`
  的默认字段自动补全不稳定
- 因此配置文件中已经显式写全关键模型字段，避免出现 `encoder=None` 之类的解析错误

### 第一版正式训练命令

先跑 `Nano`：

```bash
rfdetr fit --config configs/rfdetr_visdrone_nano.yaml
```

如果你更想直接跑 `Small`：

```bash
rfdetr fit --config configs/rfdetr_visdrone_small.yaml
```

### 常用训练过程命令

查看最新 checkpoint：

```bash
ls output/visdrone_nano
```

从上次中断处继续训练：

```bash
rfdetr fit \
  --config configs/rfdetr_visdrone_nano.yaml \
  --ckpt_path output/visdrone_nano/last.ckpt
```

用已有 checkpoint 做验证：

```bash
rfdetr validate \
  --config configs/rfdetr_visdrone_nano.yaml \
  --ckpt_path output/visdrone_nano/last.ckpt
```

### 关于 WanLab

当前代码显式支持的是 `TensorBoard` / `WandB` / `MLflow`。

因此建议现阶段先这样处理：

1. 先按当前配置跑通正式训练
2. 如果你的 `WanLab` 支持兼容 `wandb`，再把配置中的：

```yaml
wandb: false
```

改成：

```yaml
wandb: true
```

并补上对应的运行环境配置

在没有确认 `WanLab` 接入方式之前，不建议现在就把日志链路绑死到 `wandb=true`。

## Git 约定

- `git add` 时必须避免把数据集、大权重、训练输出带进版本库。
- `visdrone/` 视为本地数据目录，默认不提交数据内容。
- 大文件默认忽略，避免卡住本地提交和远程同步。

## 仓库精简原则

正式开发前，可以删除与当前目标无关的内容，重点精简：

- 文档站相关文件
- 发布展示类文件
- 当前训练开发不会用到的演示材料

但以下内容默认保留：

- `src/`
- `configs/`
- `tests/`
- `pyproject.toml`
- `.pre-commit-config.yaml`
- `AGENTS.md`

## 当前验证状态

我已经确认：

- 项目训练依赖和训练入口位置
- RF-DETR 当前需要的输入格式
- 你当前 `VisDrone` 的真实目录结构和标注形态
- 你当前 `VisDrone` 需要先转换后再训练

我当前还没有完成的只有一项：

- 本机无法直接跑完整训练 smoke，因为当前环境缺少 `torch` 和 `pytorch_lightning`

但转换脚本和环境检查脚本已经补齐，服务器环境就绪后可以直接执行。
