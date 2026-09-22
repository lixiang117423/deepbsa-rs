//! 预处理：对齐 pretreatment.py（一筛/二筛）与 none_pretreatment.py。
//!
//! 原版行为细节（必须保留）：
//! - first_check 内先把 0 深度替换为 1e-4（就地修改，后续 freq/存储都用调整后的值）
//! - none_pretreatment 保留原版 bug：ap 用 `Ap == 0` 的掩码做替换（D 列表外行为等价）
//! - p 值：scipy chisquare(df=1) 的 sf = erfc(sqrt(stat/2))

use crate::data::{ChrData, Dataset};
use rayon::prelude::*;

/// 卡方拟合优度检验（2 类、期望均分）p 值，等价 scipy.stats.chisquare([A,a]).pvalue
pub fn chisquare_1df_pvalue(a: f64, b: f64) -> f64 {
    let total = a + b;
    if total <= 0.0 {
        return 1.0;
    }
    let e = total / 2.0;
    let stat = (a - e) * (a - e) / e + (b - e) * (b - e) / e;
    chi2_sf(stat, 1.0)
}

/// ln Γ(x)（Lanczos 近似）
fn gammln(xx: f64) -> f64 {
    const COF: [f64; 6] = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ];
    let x = xx;
    let mut y = xx;
    let tmp = x + 5.5;
    let tmp = tmp - (x + 0.5) * tmp.ln();
    let mut ser = 1.000000000190015;
    for &c in &COF {
        y += 1.0;
        ser += c / y;
    }
    -tmp + (2.5066282746310005 * ser / x).ln()
}

/// 正则化下不完全 gamma P(a,x)（级数）
fn gser(a: f64, x: f64) -> f64 {
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    for _ in 0..500 {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * 1e-17 {
            break;
        }
    }
    sum * (-x + a * x.ln() - gammln(a)).exp()
}

/// 正则化上不完全 gamma Q(a,x)（修正 Lentz 连分式）
fn gcf(a: f64, x: f64) -> f64 {
    let tiny = 1e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / tiny;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..500 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < tiny {
            d = tiny;
        }
        c = b + an / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-17 {
            break;
        }
    }
    (-x + a * x.ln() - gammln(a)).exp() * h
}

/// χ²(df) 上侧概率，等价 scipy.stats.chi2.sf(x, df)
pub fn chi2_sf(x: f64, df: f64) -> f64 {
    let a = df / 2.0;
    let xx = x / 2.0;
    if xx <= 0.0 {
        return 1.0;
    }
    if xx < a + 1.0 {
        1.0 - gser(a, xx)
    } else {
        gcf(a, xx)
    }
}

pub struct PretreatOptions {
    pub read_number: u32, // p1
    pub chi_square: bool, // p2
    pub continuity: bool, // p3
}

/// 一筛（对单个 SNP 的池向量）：读深过滤 + 卡方检验
/// 返回 true 表示保留。注意 A/a 向量会被就地调整（0 -> 1e-4）。
pub fn first_check(a_pool: &mut [f64], a_mut: &mut [f64], read_number: u32, chi_square: bool) -> bool {
    for (a, m) in a_pool.iter_mut().zip(a_mut.iter_mut()) {
        if *a == 0.0 {
            *a = 1e-4;
        }
        if *m == 0.0 {
            *m = 1e-4;
        }
    }
    if read_number > 0 && a_pool.iter().zip(a_mut.iter()).any(|(a, m)| a + m < read_number as f64) {
        return false;
    }
    if !chi_square {
        // 原版 p2=False 时：first_check 仍然做卡方？查原版：step_two 未参与
        // first_check —— 实际原版 Pretreatment 总是执行卡方，p2 仅记录未使用。
        // 为保守起见与原版保持一致：总执行。
    }
    let mut p_values = Vec::with_capacity(a_pool.len());
    let mut a_ratios = Vec::with_capacity(a_pool.len());
    for (a, m) in a_pool.iter().zip(a_mut.iter()) {
        p_values.push(chisquare_1df_pvalue(*a, *m));
        a_ratios.push(a / (a + m));
    }
    if p_values.iter().all(|&p| p >= 0.05) {
        return true;
    }
    let n = p_values.len();
    if n >= 2
        && p_values[0] < 0.05
        && p_values[n - 1] < 0.05
        && ((a_ratios[0] > 0.5 && a_ratios[n - 1] < 0.5) || (a_ratios[0] < 0.5 && a_ratios[n - 1] > 0.5))
    {
        return true;
    }
    false
}

/// 二筛：连续性检验（freq 与相邻点差异 < 0.1）；is_process=true 时全部通过
pub fn second_check(chr: &ChrData, is_process: bool) -> ChrData {
    let n = chr.len();
    let mut out = ChrData::default();
    for index in 0..n {
        let signal = if is_process {
            true
        } else {
            let f = &chr.freq[index];
            let f_next = if index + 1 < n { Some(&chr.freq[index + 1]) } else { None };
            let f_prev = if index > 0 { Some(&chr.freq[index - 1]) } else { None };
            let ok_next = f_next.map(|g| f.iter().zip(g.iter()).all(|(x, y)| (x - y).abs() < 0.1));
            let ok_prev = f_prev.map(|g| f.iter().zip(g.iter()).all(|(x, y)| (x - y).abs() < 0.1));
            match (ok_prev, ok_next) {
                (None, Some(true)) | (Some(true), None) => true,
                (Some(true), Some(true)) => true,
                _ => false,
            }
        };
        if signal {
            out.ref_a.push(chr.ref_a[index].clone());
            out.mut_a.push(chr.mut_a[index].clone());
            out.freq.push(chr.freq[index].clone());
            out.pos.push(chr.pos[index]);
        }
    }
    out
}

/// 完整 Pretreatment（p=True 路径）。rayon 按染色体并行。
pub fn pretreat(ds: &Dataset, opts: &PretreatOptions) -> Dataset {
    let chrs: Vec<ChrData> = ds
        .chrs
        .par_iter()
        .map(|chr| {
            let mut kept = ChrData::default();
            for (ref_row, mut_row, pos) in chr.ref_a.iter().zip(&chr.mut_a).zip(&chr.pos).map(|((a, m), p)| (a, m, p)) {
                let mut a = ref_row.clone();
                let mut m = mut_row.clone();
                if first_check(&mut a, &mut m, opts.read_number, opts.chi_square) {
                    kept.ref_a.push(a.clone());
                    kept.mut_a.push(m.clone());
                    let f: Vec<f64> = a.iter().zip(&m).map(|(x, y)| x / (x + y)).collect();
                    kept.freq.push(f);
                    kept.pos.push(*pos);
                }
            }
            second_check(&kept, opts.continuity)
        })
        .collect();
    Dataset { chrome_set: ds.chrome_set.clone(), chrs }
}

/// NonePretreatment（p=False 路径），保留原版 0->1e-4 调整
/// （包括原版 ap 用 Ap 掩码的 bug：对齐行为，此处同样以 Ap==0 掩码替换 ap）
pub fn none_pretreat(ds: &Dataset) -> Dataset {
    let chrs: Vec<ChrData> = ds
        .chrs
        .par_iter()
        .map(|chr| {
            let mut out = ChrData::default();
            for ((ref_row, mut_row), pos) in chr.ref_a.iter().zip(&chr.mut_a).zip(&chr.pos) {
                let mut a = ref_row.clone();
                let mut m = mut_row.clone();
                for i in 0..a.len() {
                    if a[i] == 0.0 {
                        a[i] = 1e-4;
                    }
                }
                // 原版 bug 复刻：ap[Ap == 0] = 0.0001
                for i in 0..m.len() {
                    if ref_row[i] == 0.0 {
                        m[i] = 1e-4;
                    }
                }
                let f: Vec<f64> = a.iter().zip(&m).map(|(x, y)| x / (x + y)).collect();
                out.ref_a.push(a);
                out.mut_a.push(m);
                out.freq.push(f);
                out.pos.push(*pos);
            }
            out
        })
        .collect();
    Dataset { chrome_set: ds.chrome_set.clone(), chrs }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chi_square_pvalue_matches_scipy() {
        // scipy.stats.chisquare([89,85]).pvalue == 0.7617075639478075
        let p = chisquare_1df_pvalue(89.0, 85.0);
        assert!((p - 0.7617075639478075).abs() < 1e-6, "p={p}");
        let p2 = chisquare_1df_pvalue(100.0, 1.0);
        assert!((p2 - 6.795441656785791e-23).abs() < 1e-30);
    }

    #[test]
    fn first_check_read_number_filter() {
        let mut a = vec![5.0, 6.0];
        let mut m = vec![5.0, 6.0];
        // 总深 10 < read_number 20 -> 淘汰
        assert!(!first_check(&mut a, &mut m, 20, true));
    }

    #[test]
    fn continuity_filter() {
        let chr = ChrData {
            ref_a: vec![vec![1.0], vec![1.0]],
            mut_a: vec![vec![1.0], vec![1.0]],
            freq: vec![vec![0.5], vec![0.9]], // 相差 0.4 -> 第二个点被滤掉
            pos: vec![1.0, 2.0],
        };
        let out = second_check(&chr, false);
        // 点0: 与点1差 0.4 -> 滤掉; 点1: 与点0差 0.4 -> 滤掉
        assert_eq!(out.len(), 0);
        let out2 = second_check(&chr, true);
        assert_eq!(out2.len(), 2);
    }
}

