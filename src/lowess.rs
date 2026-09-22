//! LOWESS：Cleveland 局部线性加权回归（it=0，无稳健迭代），O(n log n + n·q)。
//! 与 statsmodels lowess(y, x, frac, it=0, delta=0) 数值等价；
//! 另含 R loess_as 的 AICc span 自动选择（精确 trace，见设计文档 D1）。

use rayon::prelude::*;

/// 单次 LOWESS 拟合。
/// x 必须已升序；对齐 statsmodels/Cleveland 语义：
/// q = floor(frac*n)（下限 2），窗口滑动条件 x[i] > (x[nleft]+x[nright])/2，
/// tricube 权重 w = (1-(r/h)^3)^3（r > h 记 0），h = max(x_i - x_nleft, x_nright - x_i)。
/// 返回每个 x_i 处的拟合值与杠杆值 h_ii（AICc 用）。
pub fn lowess_fit(x: &[f64], y: &[f64], frac: f64) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    if n == 0 {
        return (vec![], vec![]);
    }
    if n < 2 {
        return (vec![y[0]], vec![1.0]);
    }
    let mut q = (frac * n as f64).floor() as usize;
    if q < 2 {
        q = 2;
    }
    if q > n {
        q = n;
    }
    assert!(x.windows(2).all(|w| w[0] <= w[1]), "x must be sorted");

    // 每个点独立计算（并行），再顺序写回
    let results: Vec<(f64, f64)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let (nleft, nright) = locate_window(x, i, q);
            let xi = x[i];
            let mut h = (xi - x[nleft]).max(x[nright] - xi);
            if h <= 0.0 {
                h = 1.0; // 全部 x 相同的退化情形
            }
            // 加权线性回归
            let mut sw = 0.0;
            let mut swx = 0.0;
            let mut swy = 0.0;
            let mut swxx = 0.0;
            let mut swxy = 0.0;
            for j in nleft..=nright {
                let r = (x[j] - xi).abs();
                if r > h {
                    continue;
                }
                let t = r / h;
                let w = (1.0 - t * t * t).powi(3);
                if w <= 0.0 {
                    continue;
                }
                sw += w;
                swx += w * x[j];
                swy += w * y[j];
                swxx += w * x[j] * x[j];
                swxy += w * x[j] * y[j];
            }
            if sw <= 0.0 {
                return (y[i], 1.0);
            }
            // 拟合值 yhat = a + b*xi；平滑矩阵对角（杠杆）：
            // yhat_i = Σ_j c_j y_j，c_j = w_j * (1/sw + (xi - xbar_w)(x_j - xbar_w)/sxx_w)
            let xbar = swx / sw;
            let ybar = swy / sw;
            let sxx = swxx - sw * xbar * xbar;
            let sxy = swxy - sw * xbar * ybar;
            if sxx.abs() < 1e-300 {
                // x 全等：退化为常数拟合
                let w_i = tricube(x[i], xi, h);
                return (ybar, w_i / sw);
            }
            let b = sxy / sxx;
            let a = ybar - b * xbar;
            let fitted = a + b * xi;
            let dxi = xi - xbar;
            let w_i = tricube(x[i], xi, h);
            let c_i = w_i * (1.0 / sw + dxi * dxi / sxx);
            (fitted, c_i)
        })
        .collect();
    let fitted: Vec<f64> = results.iter().map(|r| r.0).collect();
    let leverage: Vec<f64> = results.iter().map(|r| r.1).collect();
    (fitted, leverage)
}

fn tricube(xj: f64, xi: f64, h: f64) -> f64 {
    let r = (xj - xi).abs();
    if r > h || h <= 0.0 {
        return 0.0;
    }
    let t = r / h;
    let w = (1.0 - t * t * t).powi(3);
    if w > 0.0 {
        w
    } else {
        0.0
    }
}

/// Cleveland 窗口定位：窗口大小 q，最终 nright - nleft + 1 == q
fn locate_window(x: &[f64], i: usize, q: usize) -> (usize, usize) {
    let n = x.len();
    let mut nleft = 0usize;
    let mut nright = q - 1;
    // Cleveland: 窗口向右滑动直到 x[i] 不超过窗口中线
    while nright < n - 1 && x[i] > (x[nleft] + x[nright + 1]) / 2.0 {
        nleft += 1;
        nright += 1;
    }
    // 边界收缩：保证 i 在窗口内（Cleveland lowest 处理：i 靠近端点时窗口贴边）
    if i < nleft {
        nleft = i;
        nright = (i + q - 1).min(n - 1);
    } else if i > nright {
        nright = i;
        nleft = i.saturating_sub(q - 1);
    }
    (nleft, nright)
}

/// 对齐 statsmodels lowess 的输出（拟合值序列；x 已排序时无需重排）
pub fn lowess(y: &[f64], frac: f64) -> Vec<f64> {
    let x: Vec<f64> = (0..y.len()).map(|i| i as f64).collect();
    lowess_fit(&x, y, frac).0
}

/// AICc(span)：对齐 R loess_as 的判据（设计文档 D1：精确 trace）
/// 拟合方向与原版一致：x = 数据值，y = 位置（Mb）
pub struct AiccEvaluator {
    pub signal: Vec<f64>,
    pub position: Vec<f64>,
}

impl AiccEvaluator {
    pub fn new(signal: &[f64], position_mb: &[f64]) -> Self {
        // R loess 内部按 x 排序；这里预排序一次
        let mut idx: Vec<usize> = (0..signal.len()).collect();
        idx.sort_by(|&a, &b| signal[a].partial_cmp(&signal[b]).unwrap());
        let signal_sorted: Vec<f64> = idx.iter().map(|&i| signal[i]).collect();
        let pos_sorted: Vec<f64> = idx.iter().map(|&i| position_mb[i]).collect();
        AiccEvaluator { signal: signal_sorted, position: pos_sorted }
    }

    pub fn aicc(&self, span: f64) -> f64 {
        let n = self.signal.len();
        let (fitted, trace) = if n < 7 {
            (vec![0.0; n], n as f64)
        } else {
            let (f, l) = lowess_fit(&self.signal, &self.position, span);
            let trace: f64 = l.iter().sum();
            (f, trace)
        };
        let rss: f64 = self
            .position
            .iter()
            .zip(&fitted)
            .map(|(&y, &f)| (y - f) * (y - f))
            .sum();
        let sigma2 = rss / (n as f64 - 1.0);
        if sigma2 <= 0.0 || !sigma2.is_finite() {
            return f64::INFINITY;
        }
        let denom = n as f64 - trace - 2.0;
        if denom <= 0.0 {
            return f64::INFINITY;
        }
        sigma2.ln() + 1.0 + 2.0 * (2.0 * (trace + 1.0)) / denom
    }
}

/// 黄金分割搜索最小化 AICC，span ∈ [0.05, 0.95]，对齐 R optimize 的容差
pub fn optimize_span(ev: &AiccEvaluator) -> f64 {
    let (mut a, mut b) = (0.05f64, 0.95f64);
    let tol = 1e-3;
    let invphi = (5.0f64.sqrt() - 1.0) / 2.0;
    let mut c = b - invphi * (b - a);
    let mut d = a + invphi * (b - a);
    let mut fc = ev.aicc(c);
    let mut fd = ev.aicc(d);
    while (b - a).abs() > tol {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - invphi * (b - a);
            fc = ev.aicc(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + invphi * (b - a);
            fd = ev.aicc(d);
        }
    }
    let best = if fc < fd { c } else { d };
    (best * 1000.0).round() / 1000.0 // 对齐原版 round(span, 3)
}

/// evaluate_frac 的等价入口：返回最优 span（保留 3 位小数）
pub fn evaluate_frac(signal: &[f64], position_mb: &[f64]) -> f64 {
    let ev = AiccEvaluator::new(signal, position_mb);
    optimize_span(&ev)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowess_recovers_smooth_trend() {
        let n = 100;
        let y: Vec<f64> = (0..n).map(|i| (i as f64 / 10.0).sin()).collect();
        let f = lowess(&y, 0.15);
        // 拟合值应接近原始平滑信号（相关系数 > 0.99）
        let mean_y = y.iter().sum::<f64>() / n as f64;
        let mean_f = f.iter().sum::<f64>() / n as f64;
        let cov: f64 = y.iter().zip(&f).map(|(a, b)| (a - mean_y) * (b - mean_f)).sum();
        let vy = y.iter().map(|a| (a - mean_y).powi(2)).sum::<f64>().sqrt();
        let vf = f.iter().map(|a| (a - mean_f).powi(2)).sum::<f64>().sqrt();
        assert!(cov / (vy * vf) > 0.99);
    }

    #[test]
    fn constant_series() {
        let y = vec![2.0; 50];
        let f = lowess(&y, 0.2);
        assert!(f.iter().all(|&v| (v - 2.0).abs() < 1e-9));
    }
}

