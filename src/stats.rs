//! 6 种传统统计方法，逐行对齐 Statistic_Methods.py 的实现（包括原版的
//! nan 处理与池序号约定：pool 0 = 低池，pool -1 = 高池）。

use anyhow::Result;

/// K 法：对每行 freq 做 x=[0..1] 线性拟合取 |斜率|（scipy curve_fit 线性 == OLS）
pub fn k_statistic(freq: &Vec<Vec<f64>>, num_pools: usize) -> Vec<f64> {
    let denom = (num_pools - 1) as f64;
    let x: Vec<f64> = (0..num_pools).map(|i| i as f64 / denom).collect();
    let x_bar = x.iter().sum::<f64>() / num_pools as f64;
    let sxx: f64 = x.iter().map(|&xi| (xi - x_bar) * (xi - x_bar)).sum();
    freq.iter()
        .map(|row| {
            let y_bar = row.iter().sum::<f64>() / row.len() as f64;
            let sxy: f64 = x
                .iter()
                .zip(row.iter())
                .map(|(&xi, &yi)| (xi - x_bar) * (yi - y_bar))
                .sum();
            (sxy / sxx).abs()
        })
        .collect()
}

/// ED4：4*(高池频率 - 低池频率)^4
pub fn ed4(freq: &Vec<Vec<f64>>) -> Vec<f64> {
    freq.iter()
        .map(|row| 4.0 * (row[row.len() - 1] - row[0]).powi(4))
        .collect()
}

/// SNP-index 差：|高池 - 低池|
pub fn snp_index(freq: &Vec<Vec<f64>>) -> Vec<f64> {
    freq.iter()
        .map(|row| (row[row.len() - 1] - row[0]).abs())
        .collect()
}

/// G 统计量（首末池 2x2 表）。
/// 原版 n_hat = np.sum(tabel, axis=i)[j] * np.sum(tabel, axis=j)[i] / total，
/// axis 语义随 (i,j) 交换，需逐格复刻（非常规卡方期望公式，属原版行为）。
pub fn g_statistic(freq: &Vec<Vec<f64>>) -> Vec<f64> {
    freq.iter()
        .map(|row| {
            let n1 = row[0];
            let n2 = 1.0 - row[0];
            let n3 = row[row.len() - 1];
            let n4 = 1.0 - row[row.len() - 1];
            let tabel = [[n1, n2], [n3, n4]];
            let total = n1 + n2 + n3 + n4;
            let rowsum = [n1 + n2, n3 + n4];
            let colsum = [n1 + n3, n2 + n4];
            let mut g = 0.0;
            for i in 0..2 {
                for j in 0..2 {
                    let ni = tabel[i][j];
                    // S(k) = np.sum(tabel, axis=k)：S(0)=colsum, S(1)=rowsum
                    // n_hat = S(i)[j] * S(j)[i] / total
                    let first = if i == 0 { colsum[j] } else { rowsum[j] };
                    let second = if j == 0 { colsum[i] } else { rowsum[i] };
                    let n_hat = first * second / total;
                    g += ni * (ni / n_hat).ln(); // 0*ln(0) -> nan -> 后面归 0
                }
            }
            2.0 * g
        })
        .map(|g| if g.is_nan() { 0.0 } else { g })
        .collect()
}

/// LOD 统计量（仅用前两个池：pool0 = 低池，pool1 = 高池）
pub fn lod_statistic(ref_a: &Vec<Vec<f64>>, mut_a: &Vec<Vec<f64>>) -> Vec<f64> {
    ref_a
        .iter()
        .zip(mut_a.iter())
        .map(|(r, m)| {
            let n_al = r[0];
            let n_al_mut = m[0];
            let n_ah = r[1];
            let n_ah_mut = m[1];
            let n_l = n_al_mut + n_al;
            let n_h = n_ah_mut + n_ah;
            let p_l = n_al / (n_al + n_al_mut);
            let p_h = n_ah / (n_ah_mut + n_ah);
            let t1 = n_al * p_l.log10();
            let t2 = n_al_mut * (1.0 - p_l).log10();
            let t3 = n_ah * p_h.log10();
            let t4 = n_ah_mut * (1.0 - p_h).log10();
            let t5 = (n_l + n_h) * (1.0f64 / 2.0).log10();
            let lod = t1 + t2 + t3 + t4 - t5;
            // 对齐 np.nan_to_num（0*inf 产生的 nan 归 0）
            if lod.is_nan() {
                0.0
            } else {
                lod
            }
        })
        .collect()
}

/// Ridit 分析，逐行复刻 cal_ridit（包括原版正态 p 值近似公式）。
/// 原版在调用处对 ref/mut 转置后按位点交错拼列：
/// 每个位点的输入向量 = [ref_p0, mut_p0, ref_p1, mut_p1, ...]
pub fn ridit(ref_a: &Vec<Vec<f64>>, mut_a: &Vec<Vec<f64>>) -> Vec<f64> {
    let n_pos = ref_a.len();
    let pools = ref_a[0].len();
    (0..n_pos)
        .map(|i| {
            let mut a: Vec<f64> = Vec::with_capacity(2 * pools);
            for p in 0..pools {
                a.push(ref_a[i][p]);
                a.push(mut_a[i][p]);
            }
            cal_ridit(&mut a)
        })
        .collect()
}

fn cal_ridit(a: &mut [f64]) -> f64 {
    for v in a.iter_mut() {
        if *v == 0.0 {
            *v = 1e-4;
        }
    }
    let num = a.len();
    let mut sum_total = 0.0;
    let mut sum_ref = 0.0;
    let mut sum_snp = 0.0;
    for (i, &v) in a.iter().enumerate() {
        sum_total += v;
        if i % 2 == 0 {
            sum_ref += v;
        } else {
            sum_snp += v;
        }
    }
    let mut cumsum: std::collections::HashMap<u32, f64> = std::collections::HashMap::new();
    cumsum.insert(0, 0.0);
    let mut sum: std::collections::HashMap<u32, f64> = std::collections::HashMap::new();
    let mut cumfr = 0.0;
    let mut cumfrr = 0.0;
    let mut r_ref = 0.0;
    let mut r_snp = 0.0;
    let mut i = 0usize;
    let half = num as f64 / 2.0;
    while i + 1 < num {
        // 对齐原版 m = i/2（dict 键与条件都基于 m）
        let m = (i / 2) as u32;
        let s = a[i] + a[i + 1];
        sum.insert(m, s);
        if m as f64 + 1.0 < half {
            let prev = cumsum.get(&m).copied().unwrap_or(0.0);
            cumsum.insert(m + 1, prev + s);
        }
        let cm = cumsum.get(&m).copied().unwrap_or(0.0);
        let r = (cm + s / 2.0) / sum_total;
        cumfr += s * r;
        cumfrr += s * r * r;
        r_ref += a[i] * r / sum_ref;
        r_snp += a[i + 1] * r / sum_snp;
        i += 2;
    }
    let var = (cumfrr - cumfr * cumfr / sum_total) / (sum_total - 1.0);
    let ratio = var * sum_total / (sum_ref * sum_snp);
    let n: f64 = if ratio < 0.0 {
        101.0 // 对齐 math.sqrt(negative) 触发 except -> n=101
    } else {
        match (r_ref - r_snp).abs() {
            d if d.is_finite() => {
                let sq = ratio.sqrt();
                if sq == 0.0 {
                    f64::NAN // 0/0 -> nan，与原版一致落入后续分支判断
                } else {
                    d / sq
                }
            }
            _ => 101.0,
        }
    };
    let mut p = 0.0f64;
    let an = n.abs();
    if n > 100.0 || n < -100.0 {
        p = 0.0;
    } else if n <= 100.0 && n >= 1.9 {
        for k in (1..=18).rev() {
            p = k as f64 / (an + p);
        }
        p = (-0.5 * an * an).exp() / (2.0 * std::f64::consts::PI).sqrt() / (an + p);
    } else if n >= -100.0 && n <= -1.9 {
        for k in (1..=18).rev() {
            p = k as f64 / (an + p);
        }
        p = (-0.5 * an * an).exp() / (2.0 * std::f64::consts::PI).sqrt() / (an + p);
        p = 1.0 - p;
    } else if n < 1.9 && n >= 0.0 {
        p = poly_p(an) / 2.0;
    } else if n > -1.9 && n < 0.0 {
        p = 1.0 - poly_p(an) / 2.0;
    }
    // nan 时所有比较为 false，p 保持 0 —— 与 python 分支行为一致
    if p == 0.0 {
        p += 0.01;
    }
    -p.abs().ln()
}

fn poly_p(an: f64) -> f64 {
    (1.0 + an * (0.049867347
        + an * (0.0211410061
            + an * (0.0032776263 + an * (0.0000380036 + an * (0.0000488906 + an * 0.000005383)))))
    ).powi(-16)
}

/// 分发：对齐原版 get_data 中除 DL 外的分支
pub fn compute(
    method: &str,
    freq: &Vec<Vec<f64>>,
    ref_a: &Vec<Vec<f64>>,
    mut_a: &Vec<Vec<f64>>,
    num_pools: usize,
) -> Result<Vec<f64>> {
    match method {
        "K" => Ok(k_statistic(freq, num_pools)),
        "ED4" => Ok(ed4(freq)),
        "SNP" => Ok(snp_index(freq)),
        "SmoothG" => Ok(g_statistic(freq)),
        "SmoothLOD" => Ok(lod_statistic(ref_a, mut_a)),
        "Ridit" => Ok(ridit(ref_a, mut_a)),
        m => Err(anyhow::anyhow!("unknown method {m}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn k_ols_slope() {
        let freq = vec![vec![0.0, 1.0], vec![0.2, 0.8]];
        let k = k_statistic(&freq, 2);
        assert!((k[0] - 1.0).abs() < 1e-12);
        assert!((k[1] - 0.6).abs() < 1e-12);
    }

    #[test]
    fn ed4_snp() {
        let freq = vec![vec![0.1, 0.9]];
        assert!((ed4(&freq)[0] - 4.0 * 0.8f64.powi(4)).abs() < 1e-12);
        assert!((snp_index(&freq)[0] - 0.8).abs() < 1e-12);
    }

    #[test]
    fn g_zero_when_extreme() {
        // n1=1, n3=0 -> 某些项 nan -> 归 0
        let g = g_statistic(&vec![vec![1.0, 0.0]]);
        assert_eq!(g[0], 0.0);
    }

    #[test]
    fn lod_basic() {
        let r = vec![vec![90.0, 10.0]];
        let m = vec![vec![10.0, 90.0]];
        let lod = lod_statistic(&r, &m);
        assert!(lod[0] > 20.0); // 完全分离时 LOD 很大
    }
}
