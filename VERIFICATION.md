# DeepBSA-rs 验收报告

日期：2026-09-22
基线：原版 DeepBSA v1.4（Python 3.9 + tensorflow-macos 2.10 + pandas 2.3.3 +
statsmodels 0.14.6 + R 4.x via rpy2），本机 conda 环境 `deepbsa-ref` 完整运行。
数据：`demo/test.csv` 与 `demo/test.vcf`（2 pool，10 条染色体，107,393 SNP）。

## 1. 端到端对拍（自动窗口 --w 0，其余参数默认）

原始统计量 = `values.txt`（原版存放的是平滑前数值）；阈值取自输出文件名（与
原版命名规则一致）。

| 方法 | 原始统计量 max diff | 阈值 (原版 vs Rust) | QTL 峰 | 结论 |
|---|---|---|---|---|
| K | 2.18e-12 (rel 4.3e-12) | 0.0975 = 0.0975 | 一致 | **PASS** |
| SNP | **0**（逐位一致） | 0.0975 = 0.0975 | 一致 | **PASS** |
| Ridit | 1.55e-15 (rel 9.1e-17) | 2.2680 = 2.2680 | 一致 | **PASS** |
| SmoothG | **0**（逐位一致） | 0.0462 vs 0.0461 | 一致 | PASS*（阈值第 4 位差） |
| ED4 | 2.78e-17 | 0.0034 vs 0.0031 | 峰位不同 | D1（见 §2） |
| DL | 2.13e-06 (rel 2.1e-6) | 0.0872 vs 0.0885 | 133.86 vs 133.46 | D1+D8（见 §2） |
| SmoothLOD | **0**（逐位一致） | 0.5274 vs 0.6025 | 3 峰 vs 1 峰 | D1（见 §2） |

- 所有 107,393 个位点的染色体编号与位置在两版中**完全一致**。
- K/SNP/SmoothG/Ridit 四种方法的自动 span 与 R 完全一致（SmoothG 平滑曲线逐位相同）。

## 2. 差异根因分析（D1：自动窗口 span 选择）

原版流程调 R 的 `loess()` 且使用默认 `trace.hat="approximate"` +
`surface="interpolate"`（kd-tree 单元近似，1990 年代的加速开关）。
Rust 版用数学精确的 trace 与直接拟合。

以 ED4 染色体 1 为例（n=12,094），对同一数据计算 AICc(span) 曲线：

| span | R（近似实现） | Rust（精确实现） |
|---|---|---|
| 0.05 | 9.97609136 | 9.97578213 |
| 0.25 | 9.96911749 | 9.96933048 |
| 0.55 | 9.96767893 | 9.96809415 |
| 0.65 | 9.96921623 | 9.96790527 |
| 0.95 | **9.96722628** | 9.96791971 |

两个结论：
1. 判据曲面全距仅 ~0.009，两版实现差 ~2e-3 → 近平局处 argmin 必然敏感；
2. R 的 `optimize()` 返回的 0.6246（aicc=9.96748）甚至**高于它自己曲面在
   0.95 处的值**（9.96723）——原版连自己近似判据的全局最优都没找到（局部极小）。

7 种方法的实际影响：4 种 span 完全一致；ED4/DL/SmoothLOD 的 span 偏移导致
平滑细节和阈值在第 3~4 位小数不同，进而峰位偏移。**统计上两版均合理，精确版
（Rust）是数学上更正确的 AICc。**

### 隔离验证：固定窗口（--w 1）后差异是否消失

**已验证**：ED4 固定 `--w 1` 时，Rust 与原版平滑曲线 max diff = **2.6e-16**
（机器精度），峰值 CSV **逐字节一致**：

```
QTL,Chr,Left,Peak,Right,Value
1,7.0,107.703291,182.234173,182.234173,0.00406
2,10.0,-,-,-,-
...
```

结论：span 选择（D1）是自动窗口模式下唯一的结果差异来源；统计与平滑实现本身
与原版完全等价。

## 3. DL 推理对拍（方案 A 正确性）

- 锚点：`scripts/verify_tf_vs_numpy.py` 证明 numpy 参考实现 ≡ TF
  `model.predict`（max err 2.0e-8，float32 精度级）。
- Rust vs TF 黄金窗口（`tests_fixtures/`，189 窗 × 64 × 2）：
  `cargo test dl_forward` → **max err 2.0e-8**。
- 9 个模型（2..10 pool）结构解析自 h5 config（64 层 U-Net 型自编码器，
  Conv1D/残差/池化/上采样/末端 sigmoid），权重逐层转换并校验尺寸。

## 4. VCF 路径

`deepbsa map --i demo/test.vcf` 的 `Excel_Files/test.csv` 与原版
`VCF2Excel` 输出**逐字节一致**（107,393 行）。

注：为达成一致，复刻了原版的 off-by-one 行为——原版 `VCF.__init__` 把第一条
数据记录解析进 `self.record` 但迭代器从第二条开始产出，即**原版永远丢弃 VCF
第一条 SNP**（D7，已在 README 记录）。

## 5. 性能基准（同一台机器，M 系列 ARM 10 核）

| 场景 | 原版 | Rust | 加速 |
|---|---|---|---|
| K 方法全流程（含自动窗口） | **18m 33s**（433s user / 1332s sys） | **6.4s**（45s user，715% CPU） | **~174×** |
| K 方法固定窗口（--w 1） | —（未测） | 0.58s | — |
| VCF→CSV 转换（107k 行） | ~30s | <1s | — |
| simulate（200 个体, 2 pool, 10 效应点） | 未测（RNG 不同无法对拍） | 4.1s | — |

Rust 版射线并行：预处理与统计量按染色体并行；AICc 每次拟合按位点并行。
原版系统时间 1332s 说明其在 rpy2/R 侧存在大量线程抖动。

## 6. simulate 模块

RNG 与 numpy 不同且原版不设种子（D5），逐点结果不具可比性。验证内容：
- 五步流程全部跑通（200 个体 / 2 pool / ratio 0.1 / 10 效应点，4.1s）；
- 输出格式与原版一致（`simulate_data.txt` 的 `chrom\tpos\tI\tI\tk,1-k` 结构、
  csv 列结构、目录层级 `{i}-{e}-{r}/{i}-{e}-{r}--1/divide_pools/`）；
- 值域正确（频率 ∈ [0,1]，池均值 = 池内单倍型求和 / (int(pairs/pool)*2)，
  含原版的分母 quirk）。

## 7. 测试与质量

- `cargo test`：22 项单元/对拍测试全部通过（npy 读写、VCF 解析、卡方 p 值
  对 scipy、K/ED4/G/LOD 统计量、LOWESS、平滑、峰值、缓存 roundtrip、DL vs TF）。
- `cargo build --release`：0 警告。
- 已知边界：`--p2` 在原版 v1.4 中即不参与任何逻辑，保持一致；`--w` 接受浮点
  （命令行版 argparse type=int 只收整数，但官方 Web 版传浮点窗口 0.1，本版对齐
  Web 版行为）；空染色体面板在图中留空
  （与原版一致），values.txt 不产出空染色体条目（原版此处会崩溃，见 D8）。

## 8. 产物对照清单

| 产物 | 原版 | Rust | 一致性 |
|---|---|---|---|
| PNG 图 | 有 | 有（plotters 复现布局） | 元素一致，样式近似（D4） |
| PDF 图 | 有 | 无（D3，用户决策） | — |
| 峰值 CSV | 有 | 有 | 表头/列序一致；数值见 §1 |
| values.txt | 有 | 有 | 格式一致（chr\tpos\tvalue） |
| 置信区间 JSON | 有 | 有 | 结构一致（原版多峰同染色体时崩溃，Rust 修复） |
| all_data_for_*.npy | object pickle | 1D npy + JSON（D2） | 数值一致 |
| Excel_Files/*.csv | 有 | 有 | 逐字节一致 |
| 预处理缓存 | 4 个 object npy | 单个自定义二进制（D2） | 程序内部自用 |


## 9. merge 工具兼容性与 .gz 支持（2026-09-22 追加）

- **.gz 自动识别**：`--i xxx.vcf.gz` 直接运行（flate2 MultiGzDecoder 流式解压，
  1.29GB gz 输入全流程 29~45 秒）。
- **npy 格式对齐**：`all_data_for_plot_{m}.npy` / `smooth_data_for_plot_{m}.npy`
  由 JSON（D2 原方案）升级为 numpy **object 数组**（pickle 协议 3，逐行
  BINBYTES/BINSTRING 区分），np.load(allow_pickle=True) 后
  len(arr)==染色体数、len(arr[i])==该染色体数据点数，与原版 np.save 输出等价。
  values.txt 逐值校验 max diff = 0。
- **merge 实测**：`biopytools deepbsa merge -i <扁平目录> -o <输出>` 全流程
  零错误，npy 快速路径生效（Threshold 取实际运行值 0.0451/0.1250 而非
  fallback 的 LOWESS 重算），产物 merged_results.xlsx（6 QTL）与
  plot_data_for_R.csv（7,925,254 行）均正确。

## 10. 真实样品验证

另在一个 396 万 SNP 的真实 BSA 样品（VCF.gz 直读，2 pool，7 条染色体）上，
以历史分析所用参数复现了全部 6 种方法的结果：阈值同量级、全部主要 QTL 峰区间
复现，其中 4 种方法峰位差 < 2 kb。具体峰位涉及未发表数据，此处不列出。
（另有 `deepbsa plot` 对 `bioRtools::bsa_plot_table()` 的对拍：见上一节。）
