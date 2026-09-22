# DeepBSA-rs 设计文档

日期：2026-09-21
目标：将 DeepBSA v1.4（Python+R+TensorFlow）全量改写为纯 Rust 单二进制工具，
无 Python/R/TF 运行时依赖，且性能优于原版。

## 0. 范围（用户确认）

- 全量对等：7 种统计方法（DL/K/ED4/SNP/SmoothG/SmoothLOD/Ridit）+ 数据模拟模块
- 输出：PNG 图 + 全部数据产物（peaks CSV、values.txt、置信区间 JSON）；PDF 不做
- 数值一致性为验收标准：与原版对拍，峰位/区间/阈值判定必须一致，
  允许浮点末位差异

## 1. 总体架构

单 crate 二进制 `deepbsa`，两个子命令（对应原版两个脚本）：

```
deepbsa map      --i FILE --m DL --p 1 --p1 0 --p2 1 --p3 1 --s LOWESS --w 0 --t 0
deepbsa simulate --i INDIVIDUAL --p POOLS --r RATIO --e EFF_POINTS --s SAVE_PATH
```

模块划分：

| 模块 | 对应原版 | 说明 |
|---|---|---|
| `vcf.rs` | vcf_handle.py | 流式解析，取 FORMAT 中 AD 字段的 ref/alt 深度 |
| `io.rs` / `npy.rs` | pandas/numpy | csv 自动分隔符读取；npy v1.0 f64/f32/i64 读写子集 |
| `pretreat.rs` | pretreatment.py / none_pretreatment.py | 一筛（读深+卡方）、二筛（连续性） |
| `stats.rs` | Statistic_Methods.py 各方法 | K/ED4/SNP/G/LOD/Ridit，纯数学 |
| `lowess.rs` | statsmodels lowess + R loess_as | O(n) 滑窗局部线性回归 + AICc span 优化 |
| `dl.rs` | Keras .h5 推理 | 权重预转换 + 手写 Conv1D 前向传播（方案 A） |
| `peaks.rs` | oneD_peaks_finder 等 | 阈值、峰值查找、排序 |
| `ci.rs` | confidence_interval_plot | 置信区间 JSON |
| `plot.rs` | matplotlib | plotters 复现布局 |
| `simulate.rs` | simulate_progress.py | 模拟流程五步 |
| `main.rs` | main.py / simulate_progress.py | clap CLI、目录结构、缓存 |

工作目录产物与原版一致：
`Pretreated_Files/`、`NoPretreatment/`、`Excel_Files/`、`Results/<样本名>/`。

## 2. 关键设计决策

### 2.1 DL 推理（方案 A，用户已确认）
- 9 个 `.h5`（row_finetune{2..10}pool.h5）用 h5py 一次性转换为自定义
  二进制权重文件 `weights/{N}pool.bin`（JSON 头描述层结构 + 裸 f32 数据）。
- 模型为 1D 卷积自编码器，算子仅 5 种：
  `Conv1D(k=1/k=3, 'same', relu)`、残差 `Add`、`MaxPool1D(2,stride=2)`、
  `UpSampling1D(2)`（值重复）、末端 Conv1D(sigmoid)。
- 前向传播手写 ~250 行，f32 计算（与 TF 一致），batch 内可 rayon 并行。
- 运行时按输入池数加载对应权重文件；权重目录查找顺序：
  `--weights` 参数 > 环境变量 `DEEPBSA_WEIGHTS` > 可执行文件同级 `weights/`。

### 2.2 LOWESS 与自动窗口
- statsmodels `lowess(it=0, delta=0)` ≡ Cleveland 经典局部线性加权回归
  （tricube 权重、按 frac 取最近邻、无稳健迭代）。x 为等距索引，
  用滑窗增量更新实现 O(n)，结果与直接加权最小二乘在浮点精度内一致。
- R `loess_as` 的 span 优化（AICc 最小化，span∈[.05,.95]）：用黄金分割搜索；
  AICc 中 traceL 用**精确** smoother trace（R 默认是近似值，见偏差清单 D1）。

### 2.3 性能
- VCF 流式解析（无正则、零拷贝切分）
- 预处理按染色体 rayon 并行；统计量逐染色体并行
- DL 批量推理：窗口 (64, pools) 合批成矩阵乘（k=1 卷积即 GEMM，k=3 用 im2col）
- LOWESS 滑窗 O(n)（原版 R/statsmodels 为 O(n²) 级）

## 3. 与原版的行为偏差清单（验收对照用）

| # | 偏差 | 原因 | 影响 |
|---|---|---|---|
| D1 | AICc span 优化用精确 trace.hat；R 默认为 kd-tree 近似 | 无法/不值得复刻 R 近似算法 | span 可能偏移小量；以下游峰位一致性为准 |
| D2 | 预处理中间缓存为自定义二进制；Results 下绘图数据为**numpy object npy**（自实现 pickle-3 序列化，与原版 np.save 等价） | 兼容 biopytools deepbsa merge 的 npy 快速路径 | np.load(allow_pickle=True) 逐值一致；权威数据产物 values.txt 格式不变 |
| D3 | PDF 图不输出（用户决策） | Rust 无等价 matplotlib | 数据齐全，可后处理 |
| D4 | PNG 版式用 plotters 复现，细节样式有差异 | 无 matplotlib | 内容元素一致（散点/平滑线/阈值线/面板标题） |
| D5 | simulate 模块 RNG 与 numpy 不同且原版本就不设种子 | 随机过程本身不确定 | 统计性质等价，逐点结果不比 |
| D6 | 卡方 p 值用不完全 gamma（Numerical Recipes 级数/连分式）计算 | 需要 scipy 级尾部精度 | 与 scipy 一致到 1e-14 |
| D7 | **复刻原版 VCF off-by-one bug**：迭代器从不产出第一条数据记录，原版静默丢弃 VCF 首条 SNP | 保持产物逐字节一致（1/107393 个位点，无统计影响） | VCF→CSV 与原版逐字节一致 |
| D8 | DL 前向传播用 f32 顺序累加（TF 为 SIMD 分块累加） | 浮点求和顺序不可复刻 | 窗口级对拍 max err 2e-8；端到端原始统计量 rel diff 2e-6 |

## 4. 验证方案

1. **黄金基线**：conda 建 py3.9 环境尝试运行原版（TF arm64 用 tensorflow-macos
   替代），对 demo 数据生成参考输出；R 部分用本机 Rscript 独立复算 span。
2. **模块级对拍**：每个统计方法写 python 参考脚本（numpy/scipy/statsmodels），
   与 Rust 输出比对（相对误差 < 1e-9；DL < 1e-4）。
3. **端到端**：demo/test.csv 与 test.vcf 各跑 7 种方法，检查峰位/区间一致；
   计时对比原版。
4. 结果写入 `VERIFICATION.md`。

## 5. 明确不做（YAGNI）

- Windows/macOS Intel 预编译、GUI、多语言绑定
- 并行多文件批处理（原版也没有）
- h5 直接解析（转换脚本一次生成权重即可）


## 6. 实施期修复的原版 bug（D8 补充）

| 原版缺陷 | 位置 | 本版处理 |
|---|---|---|
| Tri-kernel-smooth 引用未导入的 QApplication，调用必崩（功能死亡） | Statistic_Methods.py smooth_function | 按算法语义正确实现 |
| 同染色体多峰时 `chrome + "*"` 对 float 染色体名崩溃 → 置信区间 JSON 无法产出 | confidence_interval_plot | 键名统一字符串化后加 `*` |
| 峰值查找中最后一个超阈 run 前有间断时 append 空集 → argmax 崩溃 | oneD_peaks_finder | 按连续段语义取全部 run |
| 染色体含空数据时 values.txt 循环索引与数据数组错位 → IndexError | Statistic.run | 按染色体索引对齐（None 占位） |
| simulate: saved_pairs 文件名 1-based 而逻辑按 0-based | divide_pools/cal_and_sort | 对齐原版 1-based 文件名 |

## 7. 验收结论速览（详见 VERIFICATION.md）

- 7 种方法原始统计量全部与原版一致（SNP/SmoothG 完全一致，其余 ≤4e-12，DL 2e-6）
- K/SNP/SmoothG/Ridit 四种方法：自动 span、平滑曲线、阈值、QTL 峰全部一致
- ED4/DL/SmoothLOD：span 因 D1 偏移 → 阈值/峰位有小差异；固定 `--w` 后平滑与
  statsmodels 逐位一致
- 性能：原版 K 方法 18m33s → Rust 6.4s（174×，含 R span 优化的完整流程）
