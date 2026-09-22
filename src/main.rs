//! DeepBSA-rs：深度学习 BSA QTL 定位工具的纯 Rust 重写。
//! 子命令 map 对应原版 main.py，simulate 对应 simulate_progress.py。

mod bsaplot;
mod ci;
mod data;
mod dl;
mod fonts;
mod io;
mod lowess;
mod npy;
mod peaks;
mod plot;
mod pretreat;
mod simulate;
mod smooth;
mod stats;
mod vcf;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "deepbsa", version, about = "Deep learning based Bulk Segregant Analysis QTL mapping (Rust rewrite of DeepBSA v1.4)")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// QTL mapping（原版 main.py）
    Map(MapArgs),
    /// 数据模拟（原版 simulate_progress.py）
    Simulate(simulate::SimulateArgs),
    /// 分面可视化 + 显著位点表（对应 bioRtools::bsa_plot_table）
    Plot(bsaplot::PlotArgs),
}

#[derive(clap::Args)]
struct MapArgs {
    /// 输入文件路径 (vcf/csv)
    #[arg(long = "i", required = true)]
    input: String,
    /// 统计算法: DL/K/ED4/SNP/SmoothG/SmoothLOD/Ridit
    #[arg(long = "m", default_value = "DL")]
    method: String,
    /// 是否预处理 (1/0)
    #[arg(long = "p", default_value = "1")]
    p: u8,
    /// 一筛：最小读深（低于则过滤）
    #[arg(long = "p1", default_value = "0")]
    p1: u32,
    /// 二筛参数：卡方检验 (1/0)，原版 v1.4 中该参数实际未参与逻辑（保持兼容）
    #[arg(long = "p2", default_value = "1")]
    p2: u8,
    /// 三筛参数：连续性检验 (1/0)
    #[arg(long = "p3", default_value = "1")]
    p3: u8,
    /// 平滑方法: Tri-kernel-smooth / LOWESS / Moving Average
    #[arg(long = "s", default_value = "LOWESS")]
    s: String,
    /// 平滑窗口比例（0=自动；Tri-kernel/Moving Average 下也可为小数，如 0.1）
    #[arg(long = "w", default_value = "0")]
    w: f64,
    /// 峰值阈值 (0=自动 median+3*std)
    #[arg(long = "t", default_value = "0")]
    t: f64,
    /// 权重目录（DL 方法用；默认 DEEPBSA_WEIGHTS 或可执行文件同级 weights/）
    #[arg(long = "weights")]
    weights: Option<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Map(args) => run_map(args),
        Command::Simulate(args) => simulate::run(args),
        Command::Plot(args) => bsaplot::run(Path::new(&args.input), Path::new(&args.output)),
    }
}

fn run_map(args: MapArgs) -> Result<()> {
    let p_flag = args.p != 0;
    println!("{}", "-".repeat(100));
    println!(
        "file path:{}\nmethod:{}\nis pretreatment:{}\nread number:{}\nChi-square test:{}\nContinuity test:{}\nsmooth method:{}\nsmooth window size:{}\nthreshold:{}",
        args.input, args.method, p_flag, args.p1, args.p2 != 0, args.p3 != 0, args.s, args.w, args.t
    );
    println!("{}", "-".repeat(100));

    let root_path = std::env::current_dir()?;
    let pretreatment_dir = root_path.join("Pretreated_Files");
    let nopretreatment_dir = root_path.join("NoPretreatment");
    let excel_path = root_path.join("Excel_Files");
    for dir in [&pretreatment_dir, &nopretreatment_dir, &excel_path] {
        std::fs::create_dir_all(dir)?;
    }

    // 文件名/类型切分（对齐原版 temp.split(".")[0] / [1]）
    let temp = args.input.rsplit('/').next().unwrap_or(&args.input).to_string();
    let mut parts = temp.splitn(2, '.');
    let file_name = parts.next().unwrap_or(&temp).to_string();
    let file_type = parts.next().context("input filename must have an extension (vcf/csv)")?;

    // 载入数据
    let file_path: PathBuf = if file_type.contains("vcf") {
        let out = excel_path.join(format!("{file_name}.csv"));
        println!("vcf to excel");
        let stats = vcf::vcf_to_csv(Path::new(&args.input), &out)
            .with_context(|| format!("parse vcf {}", args.input))?;
        println!("CSV saved to: {}", out.display());
        let _ = stats;
        out
    } else {
        PathBuf::from(&args.input)
    };

    // 预处理（带缓存）
    let cache_dir = if p_flag { &pretreatment_dir } else { &nopretreatment_dir };
    let cache_stem = if p_flag {
        format!("{file_name}_{}", args.p1)
    } else {
        file_name.clone()
    };
    let cache_file = cache_dir.join(format!("{cache_stem}.bin"));
    let (dataset, cache_hit_flag) = if cache_file.exists() {
        let ds = data::load_cache(&cache_file)?;
        println!("pretreatment & files exist");
        (ds, true)
    } else {
        let table = io::read_table(&file_path)?;
        let raw = data::Dataset::from_table(&table)?;
        let ds = if p_flag {
            pretreat::pretreat(
                &raw,
                &pretreat::PretreatOptions {
                    read_number: args.p1,
                    chi_square: args.p2 != 0,
                    continuity: args.p3 != 0,
                },
            )
        } else {
            pretreat::none_pretreat(&raw)
        };
        data::save_cache(&cache_file, &ds)?;
        println!("pretreatment & files do not exist");
        (ds, false)
    };
    let _ = cache_hit_flag;
    println!("{:?}", dataset.chrome_set);

    let rsp = root_path.join("Results");
    std::fs::create_dir_all(&rsp)?;
    let save_path = rsp.join(&file_name);
    std::fs::create_dir_all(&save_path)?;

    run_statistic(&args, &dataset, &save_path)
}

/// 对齐 Statistic.run
fn run_statistic(args: &MapArgs, dataset: &data::Dataset, save_path: &Path) -> Result<()> {
    let auto_win = args.w == 0.0;
    let num_pools = dataset.n_pools();
    if num_pools == 0 {
        bail!("no data after pretreatment");
    }

    // DL 模型按需加载（一次）
    let model = if args.method == "DL" {
        let dir = dl::weights_dir(args.weights.as_ref().map(Path::new))?;
        Some(dl::DLModel::load_for_pools(&dir, num_pools)?)
    } else {
        None
    };

    let n_chr = dataset.chrs.len();
    let mut all_percentile: Vec<f64> = Vec::new();
    let mut chr_plot: Vec<Option<Vec<f64>>> = vec![None; n_chr];
    let mut chr_smooth: Vec<Option<Vec<f64>>> = vec![None; n_chr];
    let mut chr_pos_sorted: Vec<Vec<f64>> = vec![Vec::new(); n_chr];

    use rayon::prelude::*;
    // 每染色体并行计算统计量 + 平滑
    let results: Vec<Option<(Vec<f64>, Vec<f64>, f64)>> = dataset
        .chrs
        .par_iter()
        .map(|chr| -> anyhow::Result<Option<(Vec<f64>, Vec<f64>, f64)>> {
            if chr.is_empty() {
                return Ok(None);
            }
            let values = if args.method == "DL" {
                dl::dl_statistic(model.as_ref().unwrap(), &chr.freq)
            } else {
                stats::compute(&args.method, &chr.freq, &chr.ref_a, &chr.mut_a, num_pools)?
            };
            let span = if auto_win {
                let pos_mb: Vec<f64> = chr.pos.iter().map(|p| p / 1e6).collect();
                let span = lowess::evaluate_frac(&values, &pos_mb);
                println!("auto window ratio: {span}");
                span
            } else {
                args.w
            };
            let y = smooth::smooth(&values, &args.s, span, 1)?;
            Ok(Some((values, y, span)))
        })
        .collect::<Result<Vec<_>>>()?;

    for (ci, r) in results.into_iter().enumerate() {
        let Some((values, y, _span)) = r else { continue };
        let mut idx: Vec<usize> = (0..dataset.chrs[ci].len()).collect();
        let pos = &dataset.chrs[ci].pos;
        idx.sort_by(|&a, &b| pos[a].partial_cmp(&pos[b]).unwrap());
        chr_pos_sorted[ci] = idx.iter().map(|&i| pos[i] / 1e6).collect();
        chr_plot[ci] = Some(idx.iter().map(|&i| values[i]).collect());
        chr_smooth[ci] = Some(idx.iter().map(|&i| y[i]).collect());
        all_percentile.extend(idx.iter().map(|&i| values[i]));
        eprintln!("chrome: {} done", dataset.chrome_set[ci]);
    }

    let smooth_flat: Vec<f64> = chr_smooth
        .iter()
        .flatten()
        .flat_map(|v| v.iter().cloned())
        .collect();

    // 阈值
    let threshold_f = if args.t == 0.0 {
        peaks::cal_threshold(&smooth_flat)
    } else {
        args.t
    };
    let threshold_str = if args.t == 0.0 {
        format!("{:.4}", threshold_f)
    } else {
        format!("{:.2}", threshold_f)
    };
    let window_str = if auto_win { "auto".to_string() } else { format!("{}", args.w) };

    // 绘图
    let panels: Vec<plot::PanelSpec> = (0..n_chr)
        .map(|i| plot::PanelSpec {
            position: &chr_pos_sorted[i],
            scatter: chr_plot[i].as_deref().unwrap_or(&[]),
            smooth: chr_smooth[i].as_deref().unwrap_or(&[]),
            title: &dataset.chrome_set[i],
        })
        .collect();
    let ymax = all_percentile.iter().cloned().fold(0.0f64, f64::max) * 1.01;
    let png_path = save_path.join(format!(
        "{}-{}-{}-{}-{}.png",
        args.p1, args.method, args.s, window_str, threshold_str
    ));
    plot::plot(&png_path, &panels, ymax, threshold_f)
        .with_context(|| format!("plot {}", png_path.display()))?;

    // 中间数据产物：与原版同名 .npy（2D 零填充，numpy 可直接加载；
    // biopytools deepbsa merge 的 npy 快速路径依赖这两个文件）
    npy::write_f64_1d(
        &mut std::io::BufWriter::new(std::fs::File::create(
            save_path.join(format!("all_data_for_percentile_{}.npy", args.method)),
        )?),
        &all_percentile,
    )?;
    let plot_rows: Vec<Vec<f64>> = (0..n_chr)
        .map(|i| chr_plot[i].clone().unwrap_or_default())
        .collect();
    npy::write_object_ragged(
        &mut std::io::BufWriter::new(std::fs::File::create(
            save_path.join(format!("all_data_for_plot_{}.npy", args.method)),
        )?),
        &plot_rows,
    )?;
    let smooth_rows: Vec<Vec<f64>> = (0..n_chr)
        .map(|i| chr_smooth[i].clone().unwrap_or_default())
        .collect();
    npy::write_object_ragged(
        &mut std::io::BufWriter::new(std::fs::File::create(
            save_path.join(format!("smooth_data_for_plot_{}.npy", args.method)),
        )?),
        &smooth_rows,
    )?;

    // values.txt（格式与原版一致：chr\tpos\tvalue，pos 为截断整数）
    {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(
            save_path.join(format!("{} values.txt", args.method)),
        )?);
        for flag in 0..n_chr {
            let (Some(plot_data), pos) = (&chr_plot[flag], &chr_pos_sorted[flag]) else {
                continue;
            };
            for (&p, &v) in pos.iter().zip(plot_data) {
                writeln!(f, "{}\t{}\t{}", dataset.chrome_set[flag], (p * 1e6) as i64, v)?;
            }
        }
        f.flush()?;
    }

    // 峰值 CSV
    let chr_display: Vec<String> = dataset
        .chrome_set
        .iter()
        .map(|c| ci::numeric_key(c))
        .collect();
    let smooth_refs: Vec<Vec<f64>> = (0..n_chr)
        .map(|i| chr_smooth[i].clone().unwrap_or_default())
        .collect();
    let table = peaks::peaks_finder(
        &smooth_refs,
        &chr_pos_sorted,
        threshold_f,
        &dataset.chrome_set,
        &chr_display,
    );
    let csv_path = save_path.join(format!(
        "{}-{}-{}-{}-{}.csv",
        args.p1, args.method, args.s, window_str, threshold_str
    ));
    peaks::write_peaks_csv(&csv_path, &table)?;

    // 置信区间 JSON（有峰的染色体）
    let peaks_for_ci: Vec<(String, f64)> = table
        .rows
        .iter()
        .filter_map(|(_, chr, _, _, peak, _)| peak.parse::<f64>().ok().map(|p| (chr.clone(), p)))
        .collect();
    let ci_map = ci::confidence_interval(&peaks_for_ci, &smooth_refs, &chr_pos_sorted, &dataset.chrome_set)?;
    let ci_val = serde_json::Value::Object(ci_map.into_iter().collect());
    ci::write_json(
        &save_path.join(format!("{}_confidence_interval_data.json", args.method)),
        &ci_val,
    )?;

    println!("done: {}", save_path.display());
    Ok(())
}
