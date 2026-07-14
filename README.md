# AstroNet–DR25–PyTorch

这是针对本项目的现代 PyTorch 复现，网络结构以 Google Research 的
TensorFlow AstroNet 配置为准，NASA FDL 的 PyTorch 版本仅作为交叉参考。

## 在线阅读

项目网站：[Kepler系外行星机器学习](https://sutony1.github.io/astronet-dr25-pytorch/)

- 《从星光变暗到候选行星》：Kepler DR25与光变曲线入门
- 《AstroNet-DR25-PyTorch训练与架构详解》：双分支1D-CNN和M2实验结果

网站的静态HTML位于 `docs/`，两篇原始Markdown位于 `site-content/`，
可通过 `site-tools/build-site.mjs` 重新生成。

## 项目状态

本仓库包含可复现的PyTorch模型、Kepler DR25预处理、按KIC分组的数据划分、
单GPU训练和Robovetter对照代码。原始FITS、固定长度视图、模型权重和大型结果文件
不进入Git仓库，运行脚本会从用户指定的数据目录读取或生成它们。

1000颗恒星M2实验的封存测试集包含233个TCE：AstroNet准确率84.12%、
精确率89.26%、召回率81.82%、ROC-AUC 0.9185、PR-AUC 0.9316。
进一步与NASA当前公共目录核对时，模型判正的121个信号中有79个当前已确认行星、
33个仍为候选体、9个当前假阳性。该结果是回顾性核对，不代表首次发现这些行星。

## 安装

建议使用Python 3.10或更高版本：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Windows PowerShell激活环境时使用：

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

运行测试：

```bash
python -m pytest -q
```

## 目录结构

```text
src/astronet_dr25/  双分支1D-CNN、数据集、指标和预处理
scripts/            下载规划、视图构建、训练、评测和M2启动脚本
tests/              网络参数量和预处理单元测试
```

两个Shell启动脚本默认使用 `/mnt/cloud_3t5/exoplanet_data`，可用环境变量覆盖：

```bash
PROJECT_ROOT=/path/to/astronet-dr25-pytorch \
DATA_ROOT=/path/to/exoplanet_data \
bash scripts/launch_m2_1000.sh
```

## 已核对的结构

- global 输入：2001点，5个卷积块，每块2层 Conv1D
- local 输入：201点，2个卷积块，每块2层 Conv1D
- 卷积核：5；每个卷积后 ReLU；块后 valid MaxPool1D
- 拼接特征：16,576维
- 共享分类器：4层512维全连接层，最后输出1个logit
- 参数总数：9,940,193

NASA FDL 参考实现不能原样作为“完全复现”：其公开 `astronet.py` 只有3层
512维隐藏层，并且有一处卷积后缺少 ReLU。本项目按 Google 官方配置修正。

## 数据容量

- DR25 TCE：34,032
- 有 KOI 正负标签的 TCE：8,054
- 有监督训练涉及的唯一 KIC：6,923
- MAST long-cadence FITS：107,364
- 目录报告估计：46,452,401,152 bytes（约43.26 GiB）
- 舍入安全上界：46,507,371,520 bytes（约43.31 GiB）
- 下载硬上限：200 GiB

## 单GPU约束

训练命令使用 `CUDA_VISIBLE_DEVICES=0`，因此进程只看得到第一张 RTX 3090。
第二张卡不参与当前基线，以减少实验变量。

## 主要入口

- `scripts/m1_dataset_budget.py`：经验容量预算
- `scripts/m1_plan_download.py`：MAST文件级清单，不下载FITS
- `scripts/m1_download_supervised.py`：硬上限、空间检查、断点续传、SHA-256
- `scripts/m1_build_views.py`：官方样条、相位折叠、2001/201视图
- `scripts/m1_train.py`：单GPU训练
- `scripts/m1_compare_robovetter.py`：同一测试集指标与图表

## 评测限制

`koi_pdisposition` 和 `koi_score` 都是 DR25 Robovetter 体系的产物。直接比较时，
Robovetter拥有标签来源优势，因此当前对照衡量的是协议和标签一致性，不足以证明
任何模型具有独立的科学优越性。正式研究应再建立独立确认行星/人工审查测试集。
