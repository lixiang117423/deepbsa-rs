//! 平滑方法分发：对齐 smooth_function 的三种实现。
//! - Tri-kernel-smooth：原版因 QApplication 未导入实际不可用（bug），此处按算法语义正确实现（偏差 D-bugfix）
//! - LOWESS：见 lowess.rs
//! - Moving Average：np.convolve(data, ones(w)/w, 'same')

use crate::lowess;

pub fn smooth(data: &[f64], function_name: &str, window_size: f64, _step: usize) -> anyhow::Result<Vec<f64>> {
    match function_name {
        // 对齐原版：int(window_size * len(data))，窗口是比例
        "Tri-kernel-smooth" => Ok(tri_cube_kernel_regression(
            data,
            (window_size * data.len() as f64) as usize,
            1,
        )),
        "LOWESS" => Ok(lowess::lowess(data, window_size)),
        "Moving Average" => Ok(moving_average(data, window_size as usize)),
        name => Err(anyhow::anyhow!(
            "unknown smooth function: {name} (expected Tri-kernel-smooth / LOWESS / Moving Average)"
        )),
    }
}

/// Tri-cube 核加权平均，含原版奇偶窗口分支与零填充语义
pub fn tri_cube_kernel_regression(data: &[f64], window_size: usize, step: usize) -> Vec<f64> {
    let weight_cal = |x: f64| (1.0 - x * x * x).powi(3);
    let (center, left, right) = if window_size % 2 == 1 {
        let center = (window_size + 1) / 2;
        let left = center - (window_size - 1) / 2;
        let right = center + (window_size - 1) / 2;
        (center, left, right)
    } else {
        let center = window_size / 2;
        let left = center - window_size / 2 + 1;
        let right = center + window_size / 2;
        (center, left, right)
    };
    let left_offset = center - left;
    let right_offset = right - center;
    let n = data.len();
    let tail_len = right_offset + step - (n - 1) % step;
    let mut padded = vec![0.0f64; left_offset];
    padded.extend_from_slice(data);
    padded.extend(std::iter::repeat(0.0).take(tail_len));
    let padded_len = padded.len();

    let mut result = Vec::new();
    let mut c_index = center as isize - 1;
    while c_index < (padded_len - right_offset - 1) as isize {
        let c = c_index as usize;
        let lo = c - left_offset;
        let hi = c + right_offset;
        let window = &padded[lo..=hi];
        let max_dist = right_offset.max(left_offset) as f64;
        if max_dist == 0.0 {
            result.push(window[0]);
            c_index += step as isize;
            continue;
        }
        let mut num = 0.0;
        let mut den = 0.0;
        for (j, &v) in window.iter().enumerate() {
            let dist = (j as isize - left_offset as isize).abs() as f64;
            let w = weight_cal(dist / max_dist);
            num += w * v;
            den += w;
        }
        result.push(num / den);
        c_index += step as isize;
    }
    // 对齐 np.nan_to_num
    result.into_iter().map(|v| if v.is_nan() { 0.0 } else { v }).collect()
}

/// np.convolve(data, ones(w)/w, 'same')
pub fn moving_average(data: &[f64], w: usize) -> Vec<f64> {
    let n = data.len();
    let out_len = n.max(w);
    if w == 0 {
        return vec![0.0; out_len];
    }
    // full 卷积后取中心段
    let full_len = n + w - 1;
    let mut full = vec![0.0f64; full_len];
    for (i, &a) in data.iter().enumerate() {
        if a == 0.0 {
            continue;
        }
        for (j, _) in (0..w).enumerate() {
            full[i + j] += a / w as f64;
        }
    }
    let start = (full_len - out_len) / 2;
    full[start..start + out_len].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moving_average_matches_numpy() {
        // np.convolve([1,2,3],[0.5,0.5],'same') == [0.5,1.5,2.5]
        let r = moving_average(&[1.0, 2.0, 3.0], 2);
        assert_eq!(r, vec![0.5, 1.5, 2.5]);
    }

    #[test]
    fn tri_kernel_constant() {
        // 原版算法用零填充 -> 边缘输出被拉向 0，内部输出恒为常数
        let r = tri_cube_kernel_regression(&[5.0; 20], 7, 1);
        assert!((r[10] - 5.0).abs() < 1e-12, "interior={}", r[10]);
        assert!(r[0] < 5.0, "edge should dip toward zero (faithful to original)");
    }
}
