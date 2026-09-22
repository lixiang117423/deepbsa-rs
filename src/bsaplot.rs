//! bioRtools::bsa_plot_table() 的 Rust 实现：
//! 读取 merge 产出的 plot_data_for_R.csv，绘制 facet_grid(Method × Chromosome)
//! 分面图（灰点=Raw、蓝线=Smooth、红虚线=Threshold，free scales，theme_bw 风格），
//! 并输出超阈值位点表（snake_case 列名，对齐 R 版 table）。

use anyhow::{Context, Result};
use clap::Args;
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Args)]
pub struct PlotArgs {
    /// merge 产出的 plot_data_for_R.csv
    #[arg(long = "i", required = true)]
    pub input: String,
    /// 输出目录（bsa_plot.png + qtl_significant_table.csv）
    #[arg(long = "o", required = true)]
    pub output: String,
}

/// 单个分面（一个方法 × 一条染色体）的数据
struct Panel {
    chromosome: String,
    pos_mb: Vec<f64>,
    raw: Vec<f64>,
    smooth: Vec<f64>,
    threshold: f64,
}

struct Record {
    method: String,
    chromosome: String,
    position: i64,
    pos_mb: f64,
    raw: f64,
    smooth: f64,
    threshold: f64,
    above: bool,
}

fn parse_bool(s: &str) -> bool {
    matches!(s.trim(), "True" | "true" | "TRUE" | "1")
}

fn read_records(path: &Path) -> Result<Vec<Record>> {
    let mut rdr = csv::ReaderBuilder::new()
        .flexible(true)
        .from_path(path)
        .with_context(|| format!("read {}", path.display()))?;
    let header: Vec<String> = rdr
        .headers()?
        .iter()
        .map(|h| h.to_string())
        .collect();
    let idx = |name: &str| -> Result<usize> {
        header
            .iter()
            .position(|h| h == name)
            .context(format!("column {name} missing"))
    };
    let (i_m, i_c, i_p, i_pmb, i_raw, i_sm, i_thr, i_ab) = (
        idx("Method")?,
        idx("Chromosome")?,
        idx("Position")?,
        idx("Position_Mb").ok(),
        idx("Raw_Value")?,
        idx("Smooth_Value")?,
        idx("Threshold")?,
        idx("Above_Threshold")?,
    );

    let mut out = Vec::new();
    for row in rdr.records() {
        let row = row?;
        let get = |i: usize| row.get(i).unwrap_or("");
        // Position_Mb 可缺省时由 Position 换算
        let pos: f64 = get(i_p).parse().unwrap_or(0.0);
        let pos_mb = match &i_pmb {
            Some(i) if !get(*i).is_empty() => get(*i).parse().unwrap_or(pos / 1e6),
            _ => pos / 1e6,
        };
        out.push(Record {
            method: get(i_m).to_string(),
            chromosome: get(i_c).to_string(),
            position: pos as i64,
            pos_mb,
            raw: get(i_raw).parse().unwrap_or(0.0),
            smooth: get(i_sm).parse().unwrap_or(0.0),
            threshold: get(i_thr).parse().unwrap_or(0.0),
            above: parse_bool(get(i_ab)),
        });
    }
    Ok(out)
}

/// 自然排序比较（数字段按数值比较，对齐 stringr::str_sort(numeric=TRUE)）
fn natural_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    let mut ai = a.char_indices().peekable();
    let mut bi = b.char_indices().peekable();
    loop {
        match (ai.peek(), bi.peek()) {
            (None, None) => return std::cmp::Ordering::Equal,
            (None, Some(_)) => return std::cmp::Ordering::Less,
            (Some(_), None) => return std::cmp::Ordering::Greater,
            (Some(&(ai0, ac)), Some(&(bi0, bc))) => {
                if ac.is_ascii_digit() && bc.is_ascii_digit() {
                    // 取完整数字段，按数值比较
                    let a_end = a[ai0..]
                        .find(|c: char| !c.is_ascii_digit())
                        .map(|k| ai0 + k)
                        .unwrap_or(a.len());
                    let b_end = b[bi0..]
                        .find(|c: char| !c.is_ascii_digit())
                        .map(|k| bi0 + k)
                        .unwrap_or(b.len());
                    let asub = a[ai0..a_end].trim_start_matches('0');
                    let bsub = b[bi0..b_end].trim_start_matches('0');
                    let ord = asub
                        .len()
                        .cmp(&bsub.len())
                        .then_with(|| asub.cmp(bsub))
                        .then_with(|| (a_end - ai0).cmp(&(b_end - bi0)));
                    if ord != std::cmp::Ordering::Equal {
                        return ord;
                    }
                    // 数值相等：同步快进越过两个数字段
                    for _ in ai0..a_end {
                        ai.next();
                    }
                    for _ in bi0..b_end {
                        bi.next();
                    }
                } else {
                    if ac != bc {
                        return ac.cmp(&bc);
                    }
                    ai.next();
                    bi.next();
                }
            }
        }
    }
}

/// 输入行 → 分面面板（方法保序 × 染色体自然排序）
fn build_panels(records: &[Record]) -> (Vec<String>, Vec<String>, Vec<Vec<Panel>>) {
    let mut methods: Vec<String> = Vec::new();
    for r in records {
        if !methods.iter().any(|m| m == &r.method) {
            methods.push(r.method.clone());
        }
    }
    let mut chrs: Vec<String> = Vec::new();
    for r in records {
        if !chrs.iter().any(|c| c == &r.chromosome) {
            chrs.push(r.chromosome.clone());
        }
    }
    chrs.sort_by(|a, b| natural_cmp(a, b));

    let mut grid: BTreeMap<(usize, usize), Panel> = BTreeMap::new();
    for r in records {
        let mi = methods.iter().position(|m| m == &r.method).unwrap();
        let ci = chrs.iter().position(|c| c == &r.chromosome).unwrap();
        let panel = grid
            .entry((mi, ci))
            .or_insert_with(|| Panel {
                chromosome: r.chromosome.clone(),
                pos_mb: Vec::new(),
                raw: Vec::new(),
                smooth: Vec::new(),
                threshold: r.threshold,
            });
        panel.pos_mb.push(r.pos_mb);
        panel.raw.push(r.raw);
        panel.smooth.push(r.smooth);
        panel.threshold = r.threshold;
    }

    let mut grid2: Vec<Vec<Panel>> = (0..methods.len()).map(|_| Vec::new()).collect();
    for ((mi, ci), panel) in grid {
        grid2[mi].push(panel);
        let _ = ci;
    }
    for row in &mut grid2 {
        row.sort_by_key(|p| chrs.iter().position(|c| c == &p.chromosome).unwrap());
    }
    (methods, chrs, grid2)
}

// ---------------- 绘图 ----------------

const COLOR_POINT: RGBColor = RGBColor(0xcc, 0xcc, 0xcc); // ggplot grey80
const COLOR_LINE: RGBColor = RGBColor(0x00, 0x00, 0xff); // "blue"
const COLOR_HLINE: RGBColor = RGBColor(0xff, 0x00, 0x00); // "red"
const COLOR_GRID: RGBColor = RGBColor(0xe5, 0xe5, 0xe5); // theme_bw grey90
const COLOR_STRIP_BG: RGBColor = RGBColor(0xeb, 0xeb, 0xeb); // strip 灰底

use plotters::prelude::*;
use plotters::style::FontDesc;

fn font(size: f64, bold: bool) -> FontDesc<'static> {
    FontDesc::new(
        FontFamily::SansSerif,
        size,
        if bold { FontStyle::Bold } else { FontStyle::Normal },
    )
}

fn ticks(min: f64, max: f64, target: usize) -> Vec<f64> {
    if max <= min {
        return vec![min];
    }
    let raw = (max - min) / target as f64;
    let mag = 10f64.powf(raw.log10().floor());
    let norm = raw / mag;
    let step = if norm < 1.5 {
        1.0
    } else if norm < 3.5 {
        2.0
    } else if norm < 7.5 {
        5.0
    } else {
        10.0
    } * mag;
    let start = (min / step).ceil() * step;
    let mut out = Vec::new();
    let mut v = start;
    while v <= max * (1.0 + 1e-12) {
        out.push(v);
        v += step;
    }
    out
}

fn fmt_tick(v: f64) -> String {
    let v = (v * 1e6).round() / 1e6;
    if (v - v.round()).abs() < 1e-9 && v.abs() < 1e7 {
        format!("{}", v.round() as i64)
    } else {
        format!("{v}")
    }
}

/// R 风格科学计数法：15 位有效数字 + 两位数指数（如 3.30060260012033e-07）
fn fmt_sci(v: f64) -> String {
    let s = format!("{v:.14e}");
    // 拆出尾数与指数，指数补零到两位
    match s.split_once('e') {
        Some((m, e)) => {
            let exp: i32 = e.parse().unwrap_or(0);
            format!("{m}e{:03}", exp)
        }
        None => s,
    }
}

pub fn run(input: &Path, outdir: &Path) -> Result<()> {
    std::fs::create_dir_all(outdir)?;
    println!("reading {}", input.display());
    let records = read_records(input)?;
    println!("{} data points", records.len());
    anyhow::ensure!(!records.is_empty(), "no data rows");

    let (methods, chrs, grid) = build_panels(&records);
    println!(
        "methods: {:?}\nchromosomes: {:?}",
        methods, chrs
    );

    // ---- 布局（对齐 facet_grid(Method ~ Chromosome)）----
    let n_row = methods.len();
    let n_col = chrs.len();
    let left_strip = 130; // y 轴标题/刻度
    let top_pad = 60; // 染色体 strip
    let bottom_pad = 90; // x 轴刻度 + 标题
    let right_strip = 110; // 方法 strip（竖排文字横写）
    let panel_w = ((1900.0 / n_col as f64).ceil() as i32).max(380);
    let panel_h = ((1800.0 / n_row as f64).ceil() as i32).max(300);
    let w = (left_strip + panel_w * n_col as i32 + right_strip + 20) as u32;
    let h = (top_pad + panel_h * n_row as i32 + bottom_pad + 20) as u32;
    let png_path = outdir.join("bsa_plot.png");

    crate::fonts::init();
    let root = BitMapBackend::new(&png_path, (w, h)).into_drawing_area();
    root.fill(&WHITE)?;

    // facet_grid(scales="free") 语义：x 按列共享（同一染色体所有方法同 x 轴），
    // y 按行共享（同一方法所有染色体同 y 轴）
    let mut x_range: Vec<(f64, f64)> = Vec::with_capacity(n_col);
    for ci in 0..n_col {
        let mut xmin = f64::INFINITY;
        let mut xmax = f64::NEG_INFINITY;
        for panel in &grid {
            if let Some(p) = panel.get(ci) {
                xmin = xmin.min(p.pos_mb.iter().cloned().fold(f64::INFINITY, f64::min));
                xmax = xmax.max(p.pos_mb.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
            }
        }
        if !xmin.is_finite() {
            if let Some(p) = grid.first().and_then(|r| r.first()) {
                xmin = p.pos_mb.first().cloned().unwrap_or(0.0);
                xmax = p.pos_mb.last().cloned().unwrap_or(1.0);
            }
        }
        x_range.push((xmin, xmax.max(xmin)));
    }
    let mut y_range: Vec<(f64, f64)> = Vec::with_capacity(n_row);
    for ri in 0..n_row {
        let mut ymin = f64::INFINITY;
        let mut ymax = f64::NEG_INFINITY;
        for panel in &grid[ri] {
            ymin = ymin.min(panel.raw.iter().chain(&panel.smooth).cloned().fold(f64::INFINITY, f64::min));
            ymax = ymax.max(panel.raw.iter().chain(&panel.smooth).cloned().fold(f64::NEG_INFINITY, f64::max));
        }
        if !ymin.is_finite() {
            ymin = 0.0;
            ymax = 1.0;
        }
        ymin = ymin.min(0.0);
        y_range.push((ymin, ymax.max(ymin)));
    }

    for (ri, method) in methods.iter().enumerate() {
        for (ci, chr) in chrs.iter().enumerate() {
            let panel = &grid[ri][ci];
            let x0 = left_strip + panel_w * ci as i32;
            let y0 = top_pad + panel_h * ri as i32;
            let (px0, py0) = (x0 + 10, y0 + 30);
            let (px1, py1) = (x0 + panel_w - 10, y0 + panel_h - 12);
            let pw = (px1 - px0) as f64;
            let ph = (py1 - py0) as f64;

            // 5% expansion（ggplot 连续轴默认）
            let (xmin, xmax) = x_range[ci];
            let (ymin, ymax) = y_range[ri];
            let (expand_x, expand_y) = (0.05 * (xmax - xmin), 0.05 * (ymax - ymin));
            let (dmin, dmax) = (xmin - expand_x, xmax + expand_x);
            let (ymin_e, ymax_e) = (ymin - expand_y, ymax + expand_y);
            let ytop = if ymax_e > 0.0 { ymax_e } else { 1.0 };

            let map = move |x: f64, y: f64| -> (i32, i32) {
                let fx = ((x - dmin) / (dmax - dmin)).clamp(0.0, 1.0);
                let fy = ((y - ymin_e) / (ytop - ymin_e)).clamp(0.0, 1.0);
                (px0 + (fx * pw).round() as i32, py1 - (fy * ph).round() as i32)
            };

            // theme_bw: 白底 + 浅灰网格 + 黑色边框
            for t in ticks(dmin, dmax, 5) {
                let (gx, _) = map(t, 0.0);
                vline(&root, gx, py0, py1, COLOR_GRID, 1)?;
            }
            for t in ticks(ymin_e, ytop, 4) {
                let (_, gy) = map(0.0, t);
                hline(&root, px0, px1, gy, COLOR_GRID, 1)?;
            }
            rect_border(&root, px0, py0, px1, py1)?;

            // 散点（Raw）
            for (&x, &y) in panel.pos_mb.iter().zip(&panel.raw) {
                let (px, py) = map(x, y);
                root.draw(&Circle::new((px, py), 3, ShapeStyle::from(COLOR_POINT).filled()))?;
            }
            // 阈值虚线
            let (_, ty) = map(dmin, panel.threshold);
            draw_dashed(&root, px0, ty, px1, ty, &COLOR_HLINE, 3)?;
            // 平滑线
            let pts: Vec<(i32, i32)> = panel
                .pos_mb
                .iter()
                .zip(&panel.smooth)
                .map(|(&x, &y)| map(x, y))
                .collect();
            root.draw(&PathElement::new(pts, ShapeStyle::from(COLOR_LINE).stroke_width(3)))?;

            // 染色体 strip（仅顶行）
            if ri == 0 {
                root.draw(&Rectangle::new(
                    [(x0 + 8, y0 + 2), (x0 + panel_w - 10, y0 + 26)],
                    ShapeStyle::from(COLOR_STRIP_BG).filled(),
                ))?;
                root.draw(&Text::new(
                    chr.clone(),
                    (x0 + panel_w / 2 - 24, y0 + 6),
                    font(24.0, true).color(&BLACK),
                ))?;
            }
            // 方法 strip（仅最后一列）
            if ci + 1 == n_col {
                root.draw(&Rectangle::new(
                    [(x0 + panel_w - 8, y0 + 28), (x0 + panel_w + right_strip - 18, y0 + panel_h - 10)],
                    ShapeStyle::from(COLOR_STRIP_BG).filled(),
                ))?;
                root.draw(&Text::new(
                    method.clone(),
                    (x0 + panel_w + 2, y0 + panel_h / 2 - 12),
                    font(24.0, true).color(&BLACK),
                ))?;
            }
            // x 刻度（仅底行）；y 刻度（仅首列）
            let is_bottom = ri + 1 == n_row;
            if is_bottom {
                for t in ticks(dmin, dmax, 5) {
                    let (gx, _) = map(t, 0.0);
                    let label = fmt_tick(t);
                    let w_est = label.len() as i32 * 13;
                    root.draw(&Text::new(label, (gx - w_est / 2, py1 + 6), font(20.0, false).color(&BLACK)))?;
                }
            }
            for t in ticks(ymin_e, ytop, 4) {
                let (_, gy) = map(0.0, t);
                let label = fmt_tick(t);
                let w_est = label.len() as i32 * 11;
                root.draw(&Text::new(label, (px0 - 8 - w_est, gy - 10), font(20.0, false).color(&BLACK)))?;
            }
        }
    }

    // 轴标题
    root.draw(&Text::new(
        "Position (Mb)".to_string(),
        ((left_strip + panel_w * n_col as i32) / 2 - 60, (h as i32 - bottom_pad / 2 - 10) as i32),
        font(28.0, false).color(&BLACK),
    ))?;
    // y 标题竖排放最左
    for (ri, _) in methods.iter().enumerate() {
        let y0 = top_pad + panel_h * ri as i32;
        root.draw(&Text::new(
            "Value".to_string(),
            (18, y0 + panel_h / 2 - 12),
            font(24.0, false).color(&BLACK),
        ))?;
    }

    root.present().context("render png")?;
    println!("plot saved: {}", png_path.display());

    // ---- 超阈值位点表（对齐 R 版 table 列名）----
    let table_path = outdir.join("qtl_significant_table.csv");
    {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(&table_path)?);
        writeln!(f, "method,chromosome,position_bp,position_mb,raw_value,fitted_value,method_threshold,above_threshold")?;
        let mut count = 0usize;
        for r in &records {
            if r.above {
                writeln!(
                    f,
                    "{},{},{},{:.6},{},{},{:.4},{}",
                    r.method,
                    r.chromosome,
                    r.position,
                    r.pos_mb,
                    fmt_sci(r.raw),
                    fmt_sci(r.smooth),
                    r.threshold,
                    if r.above { "TRUE" } else { "FALSE" }
                )?;
                count += 1;
            }
        }
        println!("significant table: {} rows -> {}", count, table_path.display());
    }
    Ok(())
}

fn vline<DB: DrawingBackend>(area: &DrawingArea<DB, plotters::coord::Shift>, x: i32, y1: i32, y2: i32, color: RGBColor, width: u32) -> Result<()>
where
    DB::ErrorType: 'static,
{
    area.draw(&PathElement::new(vec![(x, y1), (x, y2)], ShapeStyle::from(color).stroke_width(width)))?;
    Ok(())
}

fn hline<DB: DrawingBackend>(area: &DrawingArea<DB, plotters::coord::Shift>, x1: i32, x2: i32, y: i32, color: RGBColor, width: u32) -> Result<()>
where
    DB::ErrorType: 'static,
{
    area.draw(&PathElement::new(vec![(x1, y), (x2, y)], ShapeStyle::from(color).stroke_width(width)))?;
    Ok(())
}

fn rect_border<DB: DrawingBackend>(area: &DrawingArea<DB, plotters::coord::Shift>, x0: i32, y0: i32, x1: i32, y1: i32) -> Result<()>
where
    DB::ErrorType: 'static,
{
    area.draw(&Rectangle::new(
        [(x0, y0), (x1, y1)],
        ShapeStyle::from(&BLACK).stroke_width(2),
    ))?;
    Ok(())
}

fn draw_dashed<DB: DrawingBackend>(
    area: &DrawingArea<DB, plotters::coord::Shift>,
    x1: i32,
    y1: i32,
    x2: i32,
    y2: i32,
    color: &RGBColor,
    width: u32,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    let len = (((x2 - x1).pow(2) + (y2 - y1).pow(2)) as f64).sqrt();
    if len == 0.0 {
        return Ok(());
    }
    let (dx, dy) = ((x2 - x1) as f64 / len, (y2 - y1) as f64 / len);
    let (dash, gap) = (14.0, 10.0);
    let mut t = 0.0;
    while t < len {
        let end = (t + dash).min(len);
        let p1 = (x1 + (dx * t) as i32, y1 + (dy * t) as i32);
        let p2 = (x1 + (dx * end) as i32, y1 + (dy * end) as i32);
        area.draw(&PathElement::new(vec![p1, p2], ShapeStyle::from(color).stroke_width(width)))?;
        t += dash + gap;
    }
    Ok(())
}
