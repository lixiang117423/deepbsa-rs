//! PNG 绘图，复现原版 matplotlib 布局（设计文档 D4：样式近似）：
//! - 每染色体一个并排面板，wspace=0
//! - 散点 #b8c6d5/#9fb2c6 交替，平滑线 #ffa333，阈值虚线 #33ccff
//! - 首面板保留 y 轴，其余隐藏；末面板 x 轴标 "10e6"；中间面板隐藏左右框线

use anyhow::Result;
use plotters::prelude::*;
use plotters::style::FontDesc;

pub struct PanelSpec<'a> {
    pub position: &'a [f64], // Mb，已排序
    pub scatter: &'a [f64],  // 原始统计值
    pub smooth: &'a [f64],
    pub title: &'a str,
}

const COLOR_SCATTER: [&str; 2] = ["#b8c6d5", "#9fb2c6"];
const COLOR_SMOOTH: RGBColor = RGBColor(0xff, 0xa3, 0x33);
const COLOR_THRESHOLD: RGBColor = RGBColor(0x33, 0xcc, 0xff);

const W: u32 = 6000;
const H: u32 = 1500;

fn parse_color(s: &str) -> RGBColor {
    let r = u8::from_str_radix(&s[1..3], 16).unwrap();
    let g = u8::from_str_radix(&s[3..5], 16).unwrap();
    let b = u8::from_str_radix(&s[5..7], 16).unwrap();
    RGBColor(r, g, b)
}

fn font(size: f64) -> FontDesc<'static> {
    FontDesc::new(FontFamily::SansSerif, size, FontStyle::Normal)
}

fn draw_dashed<DB>(
    area: &DrawingArea<DB, plotters::coord::Shift>,
    x1: i32,
    y1: i32,
    x2: i32,
    y2: i32,
    color: &RGBColor,
    width: u32,
) -> Result<()>
where
    DB: DrawingBackend,
    DB::ErrorType: 'static,
{
    let len = (((x2 - x1).pow(2) + (y2 - y1).pow(2)) as f64).sqrt();
    if len == 0.0 {
        return Ok(());
    }
    let (dx, dy) = ((x2 - x1) as f64 / len, (y2 - y1) as f64 / len);
    let (dash, gap) = (24.0, 16.0);
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

fn vline<DB>(area: &DrawingArea<DB, plotters::coord::Shift>, x: i32, y1: i32, y2: i32) -> Result<()>
where
    DB: DrawingBackend,
    DB::ErrorType: 'static,
{
    area.draw(&PathElement::new(vec![(x, y1), (x, y2)], ShapeStyle::from(&BLACK).stroke_width(2)))?;
    Ok(())
}

fn hline<DB>(area: &DrawingArea<DB, plotters::coord::Shift>, x1: i32, x2: i32, y: i32) -> Result<()>
where
    DB: DrawingBackend,
    DB::ErrorType: 'static,
{
    area.draw(&PathElement::new(vec![(x1, y), (x2, y)], ShapeStyle::from(&BLACK).stroke_width(2)))?;
    Ok(())
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
    // 去浮点尾码：先按 1e-6 精度取整
    let v = (v * 1e6).round() / 1e6;
    if (v - v.round()).abs() < 1e-9 && v.abs() < 1e7 {
        format!("{}", v.round() as i64)
    } else {
        // 去掉多余的 0（如 0.30000000000000004 -> 0.3）
        let s = format!("{v}");
        s
    }
}

pub fn plot(path: &std::path::Path, panels: &[PanelSpec], ymax: f64, threshold: f64) -> Result<()> {
    crate::fonts::init();
    let n = panels.len().max(1);
    let root = BitMapBackend::new(path, (W, H)).into_drawing_area();
    root.fill(&WHITE)?;

    let top_pad = 70i32;
    let bottom_pad = 110i32;
    let panel_w = (W as i32) / n as i32;
    let ymin = 0.0f64;
    let ymax = if ymax > 0.0 { ymax } else { 1.0 };

    for (i, spec) in panels.iter().enumerate() {
        let x0 = panel_w * i as i32;
        let x1 = if i + 1 == n { W as i32 } else { panel_w * (i as i32 + 1) };
        let left_inset = if i == 0 { 130 } else { 12 };
        let right_inset = if i + 1 == n { 30 } else { 0 };
        let (rx0, ry0) = (x0 + left_inset, top_pad);
        let (rx1, ry1) = (x1 - 1 - right_inset, H as i32 - 1 - bottom_pad);
        let pw = (rx1 - rx0) as f64;
        let ph = (ry1 - ry0) as f64;

        let xmin = spec.position.first().copied().unwrap_or(0.0);
        let xmax = spec.position.last().copied().unwrap_or(1.0);
        let margin = if xmax > xmin { 0.05 * (xmax - xmin) } else { 1.0 };
        let (dmin, dmax) = (xmin - margin, xmax + margin);

        let map = |x: f64, y: f64| -> (i32, i32) {
            let fx = ((x - dmin) / (dmax - dmin)).clamp(0.0, 1.0);
            let fy = ((y - ymin) / (ymax - ymin)).clamp(0.0, 1.0);
            (rx0 + (fx * pw).round() as i32, ry1 - (fy * ph).round() as i32)
        };

        // 散点（matplotlib 默认 s=36pt^2 -> 直径约 28px @300dpi）
        for (j, (&x, &y)) in spec.position.iter().zip(spec.scatter.iter()).enumerate() {
            let (px, py) = map(x, y);
            root.draw(&Circle::new((px, py), 13, ShapeStyle::from(parse_color(COLOR_SCATTER[j % 2])).filled()))?;
        }
        // 平滑线
        if !spec.smooth.is_empty() {
            let pts: Vec<(i32, i32)> = spec
                .position
                .iter()
                .zip(spec.smooth.iter())
                .map(|(&x, &y)| map(x, y))
                .collect();
            root.draw(&PathElement::new(pts, ShapeStyle::from(COLOR_SMOOTH).stroke_width(7)))?;
        }
        // 阈值虚线
        let (xl, y_thr) = map(dmin, threshold);
        let (xr, _) = map(dmax, threshold);
        draw_dashed(&root, xl, y_thr, xr, y_thr, &COLOR_THRESHOLD, 7)?;

        // 面板标题
        root.draw(&Text::new(
            spec.title.to_string(),
            ((rx0 + rx1) / 2 - 20, ry0 - 55),
            font(40.0).color(&BLACK),
        ))?;

        let is_first = i == 0;
        let is_last = i + 1 == n;
        // top/bottom 框线（原版 spines：首面板只留左线，末面板只留右线，中间无线）
        hline(&root, rx0, rx1, ry0)?;
        hline(&root, rx0, rx1, ry1)?;
        if is_first {
            vline(&root, rx0, ry0, ry1)?;
        }
        if is_last {
            vline(&root, rx1, ry0, ry1)?;
        }

        // y 轴刻度（仅首面板）
        if is_first {
            for t in ticks(ymin, ymax, 5) {
                let (_, py) = map(dmin, t);
                vline(&root, rx0 - 12, py, rx0)?;
                root.draw(&Text::new(fmt_tick(t), (rx0 - 128, py - 18), font(32.0).color(&BLACK)))?;
            }
        }
        // x 轴刻度（所有面板）
        for t in ticks(dmin, dmax, 5) {
            let (px, _) = map(t, ymin);
            vline(&root, px, ry1, ry1 + 11)?;
            let label = fmt_tick(t);
            let w_est = label.len() as i32 * 17;
            root.draw(&Text::new(label, (px - w_est / 2, ry1 + 20), font(32.0).color(&BLACK)))?;
        }
        if is_last {
            root.draw(&Text::new("10e6".to_string(), (rx1 - 110, ry1 + 68), font(36.0).color(&BLACK)))?;
        }
    }
    root.present()?;
    Ok(())
}
