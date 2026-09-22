//! 峰值查找、自动阈值，对齐 oneD_peaks_finder / peaks_finder / cal_threshold。

use anyhow::Result;

/// 单染色体峰值查找。返回 None 表示该染色体最大值低于阈值（原版输出 "-" 行）。
/// 修复说明：原版在“最后一个超阈值 run 前有间断”时会 append 空集导致 argmax 崩溃，
/// 此处按语义取所有以连续段为单位的结果（设计文档 D-bugfix）。
pub fn one_d_peaks_finder(
    smooth_value: &[f64],
    position: &[f64],
    height: f64,
) -> Option<Vec<(f64, f64, f64, f64)>> {
    let max_v = smooth_value.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_v < height {
        return None;
    }
    let sets: Vec<usize> = (0..smooth_value.len())
        .filter(|&i| smooth_value[i] >= height)
        .collect();
    if sets.is_empty() {
        return None;
    }
    // 切分连续段
    let mut runs: Vec<Vec<usize>> = vec![vec![sets[0]]];
    for &i in &sets[1..] {
        if i == *runs.last().unwrap().last().unwrap() + 1 {
            runs.last_mut().unwrap().push(i);
        } else {
            runs.push(vec![i]);
        }
    }
    let mut out = Vec::with_capacity(runs.len());
    for run in &runs {
        let mut maxi = run[0];
        for &i in &run[1..] {
            if smooth_value[i] > smooth_value[maxi] {
                maxi = i;
            }
        }
        let left = position[run[0]];
        let right = position[*run.last().unwrap()];
        let peak = position[maxi];
        // 对齐 round(v, 5)：python 银行家舍入
        let value = round_half_even(smooth_value[maxi], 5);
        out.push((left, peak, right, value));
    }
    Some(out)
}

/// 全部染色体的峰值表 + CSV 行
pub struct PeakTable {
    /// (qtl_id, chr, left, peak, right, value)；chr 已格式化（数值染色体 -> "1.0" 风格）
    pub rows: Vec<(usize, String, String, String, String, String)>,
}

pub fn peaks_finder(
    smooth_data: &[Vec<f64>],
    position: &[Vec<f64>],
    height: f64,
    chrome_set: &[String],
    chr_display: &[String],
) -> PeakTable {
    let mut all: Vec<(usize, String, String, String, String)> = Vec::new();
    for (ci, (chr_smooth, pos)) in smooth_data.iter().zip(position.iter()).enumerate() {
        match one_d_peaks_finder(chr_smooth, pos, height) {
            Some(peaks) => {
                for (left, peak, right, value) in peaks {
                    all.push((ci, fmt(left), fmt(peak), fmt(right), fmt(value)));
                }
            }
            None => {
                all.push((ci, "-".into(), "-".into(), "-".into(), "-".into()));
            }
        }
    }
    // QTL 编号 = 拼接顺序（对齐原版：编号在排序前生成）
    let qtl: Vec<usize> = (1..=all.len()).collect();
    // 按峰值降序（"-" 视为最小）；并列时按下标降序——复刻原版
    // np.argsort()[::-1] 的反转语义（dash 行因此呈倒序）
    let mut order: Vec<usize> = (0..all.len()).collect();
    order.sort_by(|&a, &b| {
        let va = all[a].4.parse::<f64>().unwrap_or(f64::NEG_INFINITY);
        let vb = all[b].4.parse::<f64>().unwrap_or(f64::NEG_INFINITY);
        vb.partial_cmp(&va)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.cmp(&a))
    });
    let mut rows = Vec::with_capacity(all.len());
    for (rank, &i) in order.iter().enumerate() {
        let (ci, l, p, r, v) = &all[i];
        rows.push((qtl[rank], chrome_set[*ci].clone(), chr_display[*ci].clone(), l.clone(), p.clone(), r.clone(), v.clone()));
    }
    let rows = rows
        .into_iter()
        .map(|(q, _, chr, l, p, r, v)| (q, chr, l, p, r, v))
        .collect();
    PeakTable { rows }
}

pub fn write_peaks_csv(path: &std::path::Path, table: &PeakTable) -> Result<()> {
    crate::io::write_csv(
        path,
        &["QTL", "Chr", "Left", "Peak", "Right", "Value"],
        &table
            .rows
            .iter()
            .map(|(q, c, l, p, r, v)| {
                vec![q.to_string(), c.clone(), l.clone(), p.clone(), r.clone(), v.clone()]
            })
            .collect::<Vec<_>>(),
    )
}

/// cal_threshold：median + 3*std(ddof=0)，若所有点低于该值则用 90 分位；
/// 最终按 0.0001 HALF_UP 量化
pub fn cal_threshold(data: &[f64]) -> f64 {
    let std = population_std(data);
    let med = median(data);
    let mut threshold = med + 3.0 * std;
    let rounded = round_half_even(threshold, 4);
    if data.iter().all(|&v| v < rounded) {
        threshold = percentile_linear(data, 90.0);
    }
    round_half_up(threshold, 4)
}

pub fn median(data: &[f64]) -> f64 {
    let mut v = data.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.is_empty() {
        return 0.0;
    }
    match v.len() % 2 {
        0 => (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2.0,
        _ => v[v.len() / 2],
    }
}

pub fn population_std(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let var = data.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / data.len() as f64;
    var.sqrt()
}

/// np.percentile 线性插值法
pub fn percentile_linear(data: &[f64], q: f64) -> f64 {
    let mut v = data.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.is_empty() {
        return 0.0;
    }
    if v.len() == 1 {
        return v[0];
    }
    let idx = (v.len() - 1) as f64 * q / 100.0;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        v[lo]
    } else {
        v[lo] + (v[hi] - v[lo]) * (idx - lo as f64)
    }
}

/// python round()（十进制银行家舍入，按 10^d 缩放）
pub fn round_half_even(x: f64, d: i32) -> f64 {
    let s = 10f64.powi(d);
    let scaled = x * s;
    let f = scaled.floor();
    let diff = scaled - f;
    let r = if diff > 0.5 {
        f + 1.0
    } else if diff < 0.5 {
        f
    } else if (f as i64) % 2 == 0 {
        f
    } else {
        f + 1.0
    };
    r / s
}

/// Decimal.quantize ROUND_HALF_UP（half away from zero）
pub fn round_half_up(x: f64, d: i32) -> f64 {
    let s = 10f64.powi(d);
    let scaled = x * s;
    let r = if scaled >= 0.0 { (scaled + 0.5).floor() } else { (scaled - 0.5).ceil() };
    r / s
}

/// 浮点最短表示（对齐 python repr/pandas to_csv 风格）
pub fn fmt(v: f64) -> String {
    if v == v.trunc() && v.abs() < 1e15 {
        format!("{:.1}", v)
    } else {
        let s = format!("{v}");
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peaks_basic() {
        let smooth = vec![0.0, 0.5, 0.9, 0.4, 0.0, 0.0, 0.7, 0.2];
        let pos: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let r = one_d_peaks_finder(&smooth, &pos, 0.3).unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!(r[0].1, 2.0); // 峰位
        assert_eq!(r[1].1, 6.0);
    }

    #[test]
    fn peaks_below_threshold() {
        let smooth = vec![0.1, 0.2];
        assert!(one_d_peaks_finder(&smooth, &[0.0, 1.0], 0.3).is_none());
    }

    #[test]
    fn threshold_semantics() {
        // median + 3*std
        let data = vec![0.1, 0.12, 0.09, 0.11, 0.5, 0.7];
        let t = cal_threshold(&data);
        assert!(t > 0.0);
    }
}
