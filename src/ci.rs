//! 置信区间数据导出，对齐 confidence_interval_plot：
//! 以峰位为中心向两侧按 10kb 步进收集区间均值，输出 JSON。
//! 原版产物结构：{"<chr>": {"data": [...], "pos": [...], "peak": ...}}

use anyhow::Result;
use std::collections::BTreeMap;

/// peaks: (chr_display, peak_mb)；smooth_data/position_mb 为每染色体按位置排序的序列
pub fn confidence_interval(
    peaks: &[(String, f64)],
    smooth_data: &[Vec<f64>],
    position_mb: &[Vec<f64>],
    chrome_set: &[String],
) -> Result<BTreeMap<String, serde_json::Value>> {
    let mut out: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    for (chr_display, peak_mb) in peaks {
        let peak = *peak_mb * 1e6; // 对齐原版 float(peak) * 1e6
        let index = chrome_set
            .iter()
            .position(|c| numeric_key(c) == *chr_display || c == chr_display)
            .unwrap_or(0);
        let chr_data = &smooth_data[index];
        let chr_pos: Vec<f64> = position_mb[index].iter().map(|p| p * 1e6).collect();
        if chr_pos.is_empty() {
            continue;
        }
        // 左侧行走顺序为位置递减，输出前反转 -> 升序
        let mut left = collect_side(&chr_pos, chr_data, peak, true);
        left.reverse();
        let right = collect_side(&chr_pos, chr_data, peak, false);

        let mut key = chr_display.clone();
        while out.contains_key(&key) {
            key.push('*'); // 同染色体多位点：对齐原版加 * 规则
        }
        let pos: Vec<f64> = left.iter().map(|(p, _)| *p).chain(right.iter().map(|(p, _)| *p)).collect();
        let data: Vec<f64> = left.iter().map(|(_, d)| *d).chain(right.iter().map(|(_, d)| *d)).collect();
        out.insert(
            key,
            serde_json::json!({ "data": data, "pos": pos, "peak": peak }),
        );
    }
    Ok(out)
}

/// left=true：ptr 从 peak 起步进 -1e4，收集区间 (ptr, ptr+1e4)
/// left=false：ptr 步进 +1e4，收集区间 (ptr-1e4, ptr)
fn collect_side(chr_pos: &[f64], chr_data: &[f64], peak: f64, left: bool) -> Vec<(f64, f64)> {
    let mut ptr = peak;
    let mut collected = Vec::new();
    let pos0 = chr_pos[0];
    let pos_last = *chr_pos.last().unwrap();
    loop {
        let (lo_bound, hi_bound) = if left {
            ptr -= 1e4;
            if ptr < pos0 {
                break;
            }
            (ptr, ptr + 1e4)
        } else {
            ptr += 1e4;
            if ptr > pos_last {
                break;
            }
            (ptr - 1e4, ptr)
        };
        let lo = chr_pos.partition_point(|&p| p <= lo_bound);
        let hi = chr_pos.partition_point(|&p| p < hi_bound);
        if hi > lo {
            let seg = &chr_data[lo..hi];
            let mean = seg.iter().sum::<f64>() / seg.len() as f64;
            let mean_pos = chr_pos[lo..hi].iter().sum::<f64>() / (hi - lo) as f64;
            collected.push((mean_pos, mean));
        }
    }
    collected
}

/// "1" -> "1.0"（对齐原版 np 数组转 str 的显示：python str(7.0) == "7.0"）
pub fn numeric_key(chrome: &str) -> String {
    match chrome.parse::<f64>() {
        Ok(v) => {
            if v == v.trunc() && v.abs() < 1e15 {
                format!("{v:.1}")
            } else {
                format!("{v}")
            }
        }
        Err(_) => chrome.to_string(),
    }
}

pub fn write_json(path: &std::path::Path, v: &serde_json::Value) -> Result<()> {
    std::fs::write(path, serde_json::to_string(v)?)?;
    Ok(())
}
