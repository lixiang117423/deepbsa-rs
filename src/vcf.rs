//! VCF 流式解析，行为对齐 vcf_handle.py：
//! 提取 CHROM/POS/REF/ALT + 每个样本 FORMAT 中 AD 字段的 ref/alt 深度。
//! 坏行跳过并计数（对齐原版 try/except 行为）。

use anyhow::{Context, Result};
use flate2::read::MultiGzDecoder;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;

pub struct VcfRow {
    pub chrom: String,
    pub pos: i64,
    pub reference: String,
    pub alt: String,
    /// 每个样本的 (ref_depth, alt_depth)
    pub depths: Vec<(i64, i64)>,
}

pub struct VcfStats {
    pub lines_processed: usize,
    pub error_count: usize,
}

/// 按扩展名打开 VCF：.gz 自动解压（MultiGzDecoder 支持多成员流）
fn open_vcf(path: &Path) -> Result<Box<dyn Read>> {
    let f = std::fs::File::open(path)?;
    if path.extension().map(|e| e == "gz").unwrap_or(false) {
        Ok(Box::new(MultiGzDecoder::new(f)))
    } else {
        Ok(Box::new(f))
    }
}

/// 解析 VCF 并写出与原版 VCF2Excel 相同格式的 CSV：
/// chrom,pos,REF,ALT,A1,a1,A2,a2,...
pub fn vcf_to_csv(vcf_path: &Path, csv_path: &Path) -> Result<VcfStats> {
    let mut reader = BufReader::new(open_vcf(vcf_path).with_context(|| format!("open {}", vcf_path.display()))?);
    let mut out = std::io::BufWriter::new(std::fs::File::create(csv_path)?);

    let mut line = String::new();
    // 跳过头部
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            // 全是头部的空文件
            return Ok(VcfStats { lines_processed: 0, error_count: 0 });
        }
        if !line.starts_with('#') {
            break;
        }
    }

    let mut stats = VcfStats { lines_processed: 0, error_count: 0 };
    // 复刻原版 off-by-one 行为：VCF.__init__ 把第一条数据行解析进 self.record
    // 但迭代器从第二条开始产出 —— 原版永远静默丢弃第一条记录（偏差文档 D7）。
    // 注意：该行已在上面的头部循环中读入 `line`，这里直接跳过不写。
    line.clear();
    if reader.read_line(&mut line)? == 0 {
        return Ok(stats);
    }
    loop {
        let trimmed = line.trim_end_matches(['\n', '\r']);
        if !trimmed.is_empty() {
            match parse_record(trimmed) {
                Ok(row) => {
                    write_row(&mut out, &row)?;
                    stats.lines_processed += 1;
                }
                Err(_) => {
                    stats.error_count += 1;
                    if stats.error_count <= 5 {
                        println!("Warning: Skipping malformed line");
                    }
                }
            }
        }
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
    }
    out.flush()?;
    println!(
        "Total lines processed: {}, Errors: {}",
        stats.lines_processed, stats.error_count
    );
    Ok(stats)
}

fn parse_record(line: &str) -> Result<VcfRow> {
    let info: Vec<&str> = line.split('\t').collect();
    if info.len() < 10 {
        return Err(anyhow::anyhow!("too few columns"));
    }
    let pos: i64 = info[1].parse().map_err(|_| anyhow::anyhow!("bad POS"))?;
    let format: Vec<&str> = info[8].split(':').collect();
    let ad_idx = format
        .iter()
        .position(|&k| k == "AD")
        .ok_or_else(|| anyhow::anyhow!("no AD field"))?;

    let mut depths = Vec::with_capacity(info.len() - 9);
    for sample in &info[9..] {
        let fields: Vec<&str> = sample.split(':').collect();
        let ad = fields
            .get(ad_idx)
            .ok_or_else(|| anyhow::anyhow!("AD missing"))?;
        let mut it = ad.split(',');
        let ref_d: i64 = it
            .next()
            .and_then(|v| v.parse().ok())
            .ok_or_else(|| anyhow::anyhow!("bad AD ref"))?;
        let alt_d: i64 = it
            .next()
            .and_then(|v| v.parse().ok())
            .ok_or_else(|| anyhow::anyhow!("bad AD alt"))?;
        depths.push((ref_d, alt_d));
    }
    Ok(VcfRow {
        chrom: info[0].to_string(),
        pos,
        reference: info[3].to_string(),
        alt: info[4].to_string(),
        depths,
    })
}

fn write_row<W: Write>(w: &mut W, row: &VcfRow) -> Result<()> {
    // 对齐 pandas to_csv：数字原样、无引号
    let mut parts: Vec<String> = Vec::with_capacity(4 + row.depths.len() * 2);
    parts.push(row.chrom.clone());
    parts.push(row.pos.to_string());
    parts.push(row.reference.clone());
    parts.push(row.alt.clone());
    for (r, a) in &row.depths {
        parts.push(r.to_string());
        parts.push(a.to_string());
    }
    writeln!(w, "{}", parts.join(","))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_basic() {
        let line = "1\t206542\t.\tG\tA\t22948.3\tPASS\tAC=8;AF=0.5\tGT:AD:DP:GQ:PL\t0/1:64,87:151:99:x\t0/1:49,45:94:99:y";
        let r = parse_record(line).unwrap();
        assert_eq!(r.chrom, "1");
        assert_eq!(r.pos, 206542);
        assert_eq!(r.depths, vec![(64, 87), (49, 45)]);
    }

    #[test]
    fn parse_missing_ad_fails() {
        let line = "1\t5\t.\tG\tA\t1\tPASS\t.\tGT:DP\t0/1:10";
        assert!(parse_record(line).is_err());
    }
}
