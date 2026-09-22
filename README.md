# DeepBSA-rs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Language](https://img.shields.io/badge/language-Rust-orange.svg)](https://www.rust-lang.org/)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS-lightgrey.svg)]()

**English** | [中文](#中文说明)

A pure-Rust reimplementation of [DeepBSA](https://github.com/lizhao007/DeepBSA) —
deep-learning-based bulked segregant analysis (BSA) for QTL mapping — as a single
static binary with **zero Python / R / TensorFlow runtime dependencies**.

> This is an independent reimplementation of the method published in
> [Li et al., *Molecular Plant* 2022](https://doi.org/10.1016/j.molp.2022.08.004).
> It is **not** affiliated with or endorsed by the original authors.
> The pretrained network weights are **not** redistributed here — see
> [Preparing the model weights](#preparing-the-model-weights--准备模型权重).

---

## 中文说明

DeepBSA v1.4 的纯 Rust 重写版：深度学习联合分析（BSA）QTL 定位工具，编译后为
**单个二进制文件，运行时零依赖**（无需 Python、R、TensorFlow）。

| | 原版 (Python) | DeepBSA-rs |
|---|---|---|
| 运行时依赖 | Python 3.7+ / TensorFlow / R / rpy2 / pandas… | **无** |
| demo 数据单方法耗时（K，10.7 万 SNP） | 18 min 33 s | **6.4 s（约 170×）** |
| 输入格式 | VCF / CSV | VCF / **VCF.gz（自动解压）** / CSV |
| DL 推理 | TensorFlow Keras | 手写 Conv1D 前向传播（与 TF 对拍误差 2e-8） |

功能与原版全量对齐：7 种统计方法（**DL / K / ED4 / SNP / SmoothG / SmoothLOD /
Ridit**）+ 数据模拟模块 + 分面可视化，输出产物格式兼容原版与
`biopytools deepbsa merge` 工作流。

## 特性

- **单二进制**：编译后直接运行，适合 HPC/集群无 root 环境部署
- **快**：rayon 并行（按染色体/按位点），LOWESS 滑窗实现，自动窗口搜索整体加速约 2 个数量级
- **`.vcf.gz` 直接读取**：流式解压，无需预处理
- **7 种统计方法**：DL（1D 卷积自编码器，2–10 pool 各一个模型）与 6 种经典方法
- **平滑与自动窗口**：LOWESS / Tri-kernel-smooth / Moving Average；`--w 0` 时用
  精确 AICc 自动选窗（替代原版 R loess 的近似模式）
- **格式兼容**：输出文件名、目录结构、峰值 CSV / `values.txt` 与原版一致；
  `all_data_for_plot_*.npy` 等为 numpy 可直接加载的 object 数组，
  `biopytools deepbsa merge` 可走 npy 快速路径
- **可视化**：内置 `deepbsa plot`（对应 `bioRtools::bsa_plot_table()` 的分面图与显著位点表）

## 构建 / Install

**预编译二进制**（推荐）：从 [Releases](https://github.com/lixiang117423/deepbsa-rs/releases)
下载对应平台压缩包（Linux x86_64 为 musl 静态链接，任意发行版直接运行；
macOS Apple Silicon / Intel 各一个），解压即用。包内不含权重，见下节。

源码编译：

```bash
git clone https://github.com/lixiang117423/deepbsa-rs.git
cd deepbsa-rs
cargo build --release
# 产物: target/release/deepbsa
```

Linux 服务器上同样只需 `cargo build --release`（纯 Rust，无 C 库依赖）。
Rust 安装：`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`

## Preparing the model weights / 准备模型权重

因版权原因本仓库**不包含**预训练权重（原始模型版权归 DeepBSA 作者所有）。
请先从官方渠道获取 DeepBSA（[下载页](http://zeasystemsbio.hzau.edu.cn/tools.html) /
[GitHub](https://github.com/lizhao007/DeepBSA)），然后用随附脚本一次性转换：

```bash
pip install h5py
python scripts/convert_weights.py <DeepBSA目录>/bin/Models weights/
# 生成 weights/{2..10}pool.bin
```

运行时按输入池数自动加载对应模型，权重目录查找顺序：
`--weights` 参数 > 环境变量 `DEEPBSA_WEIGHTS` > 可执行文件同级 `weights/`。

## 用法 Usage

参数与原版 `main.py` / `simulate_progress.py` 一致（额外多一个 `plot` 子命令）：

```bash
# QTL mapping（默认一次跑完 7 种方法）
deepbsa map --i input.vcf.gz --p 0 --s Tri-kernel-smooth --w 0.1

# 参数（默认值同原版，--m 除外）
#   --m  all | 逗号组合（如 DL,K）              默认 all 运行全部 7 种
#   --p  1                                      是否预处理（1/0）
#   --p1 0    --p2 1    --p3 1                  三步预处理参数
#   --s  LOWESS                                 Tri-kernel-smooth | LOWESS | Moving Average
#   --w  0                                      平滑窗口比例（0=自动 AICc；可为小数）
#   --t  0                                      峰值阈值（0=自动 median+3*std）
#   --weights DIR                               权重目录（DL 方法）

# 数据模拟
deepbsa simulate --i 200 --p 2 --r 0.1 --e 10 --s out_dir

# 可视化（对应 bioRtools::bsa_plot_table()）
deepbsa plot --i merged_results/plot_data_for_R.csv --o plot_out/
```

输出目录结构与原版一致：`Excel_Files/`、`Pretreated_Files/`、`NoPretreatment/`、
`Results/<样本名>/`（PNG 图、峰值 CSV、`values.txt`、置信区间 JSON、统计量 npy）。

### 兼容性 Compatibility

- **`biopytools deepbsa merge`**：`all_data_for_plot_{m}.npy` /
  `smooth_data_for_plot_{m}.npy` / `{m} values.txt` 均为 numpy 原生格式，
  merge 走 npy 快速路径（读取真实平滑值），全流程零错误。
- **`bioRtools::bsa_plot_table()`**：`deepbsa plot` 是其 Rust 复刻——
  `facet_grid(Method ~ Chromosome)` 分面图（theme_bw 风格：灰点 Raw、蓝线
  Smooth、红虚线 Threshold，free scales，染色体自然排序）+ 超阈值位点表
  （列名对齐 R 版）。790 万数据点实测 3.2 秒，table 与 R 版逐行一致
  （R 版 `position_bp` 列实为 Mb 值，系 R 代码笔误，本版按列名语义输出真实 bp）。

## 验证 / Validation

对拍基准：原版 DeepBSA v1.4 完整运行（TensorFlow 2.10 + R），demo 数据
107,393 SNP × 10 染色体；另在 396 万 SNP 真实 BSA 样品上复现了历史分析结果。
详细数据见 [`VERIFICATION.md`](VERIFICATION.md)。摘要：

- 7 种方法原始统计量全部一致（SNP/SmoothG 逐位一致；K/Ridit/ED4 ≤ 4e-12；DL 2e-6）
- K / SNP / SmoothG / Ridit：自动窗口、平滑曲线、阈值、QTL 峰全部一致
- ED4 / DL / SmoothLOD 的自动窗口与 R 存在固有近似差异（见下），固定 `--w` 后平滑
  曲线与原版在机器精度内一致
- DL 前向传播 vs TF：`cargo test`，max err 2e-8
- VCF→CSV 转换与原版逐字节一致

### 与原版的行为差异 / Known deviations

| # | 差异 | 说明 |
|---|---|---|
| D1 | 自动窗口 AICc 用精确 trace | 原版 R loess 默认近似模式（判据曲面自带 ~2e-3 噪声）；ED4/DL/SmoothLOD 的自动窗口可能偏移，固定 `--w` 则完全一致 |
| D2 | 中间缓存/绘图数据格式 | 自定义二进制与 numpy object npy（数值一致） |
| D3 | 不输出 PDF | 数据产物齐全 |
| D5 | simulate RNG | 随机过程，统计性质等价 |
| D7 | 复刻原版 VCF 首行丢弃行为 | 保持产物逐字节一致 |
| D8 | DL 用 f32 顺序累加 | 窗口级对拍 2e-8 |

另修复了原版若干崩溃路径 bug（同染色体多峰置信区间导出、Tri-kernel-smooth
未定义符号、峰值查找空集等），完整清单见
[`docs/design.md`](docs/design.md)。

## 引用 / Citation

方法出处（本工具为其独立重实现）：

> Li Z, Chen X, Shi S, Zhang H, Wang X, Chen H, Li W, Li L.
> **DeepBSA: A deep-learning algorithm improves bulked segregant analysis for
> dissecting complex traits.**
> *Molecular Plant* 2022;15(9):1418-1427. doi: [10.1016/j.molp.2022.08.004](https://doi.org/10.1016/j.molp.2022.08.004)

原版工具：[GitHub](https://github.com/lizhao007/DeepBSA) ·
[下载页](http://zeasystemsbio.hzau.edu.cn/tools.html)

## 许可证 / License

本仓库代码以 [MIT](LICENSE) 许可发布。

DeepBSA 原版工具及其预训练模型版权归原作者所有；本仓库不分发原版代码、
模型权重或测试数据。使用 `scripts/convert_weights.py` 转换权重时，请遵守
原版的获取条款，并自行引用上述论文。

## 致谢 / Acknowledgements

感谢 DeepBSA 原作者（华中农业大学）发表的方法与工具。本项目与
[bioRtools](https://github.com/lixiang117423/bioRtools)、
[biopytools](https://github.com/lixiang117423/biopytools) 生态兼容。
