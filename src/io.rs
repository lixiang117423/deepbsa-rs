//! CSV/文本表格读取，行为对齐 pandas read_csv(header=None, sep=None, engine='python')：
//! 自动嗅探分隔符；数值列解析为数字。

use anyhow::{Context, Result};
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Table {
    /// 每行的原始字段（字符串形式，与 pandas 读入后按需转换一致）
    pub rows: Vec<Vec<String>>,
}

impl Table {
    pub fn n_cols(&self) -> usize {
        self.rows.first().map(|r| r.len()).unwrap_or(0)
    }

    #[allow(dead_code)] // 单元测试使用
    pub fn col_f64(&self, idx: usize) -> Result<Vec<f64>> {
        self.rows
            .iter()
            .map(|r| {
                let v = r
                    .get(idx)
                    .context(format!("row too short for col {idx}"))?;
                v.parse::<f64>()
                    .with_context(|| format!("parse col {idx} value {v:?}"))
            })
            .collect()
    }

    #[allow(dead_code)]
    pub fn col_str(&self, idx: usize) -> Vec<String> {
        self.rows.iter().map(|r| r.get(idx).cloned().unwrap_or_default()).collect()
    }
}

fn sniff_separator(first_line: &str) -> char {
    let candidates = [',', '\t', ';', ' '];
    let mut best = ',';
    let mut best_n = 0usize;
    for &c in &candidates {
        let n = first_line.split(c).count().saturating_sub(1);
        if n > best_n {
            best = c;
            best_n = n;
        }
    }
    best
}

/// 读取无表头表格，自动嗅探分隔符（对齐原版 pd.read_csv(sep=None)）
pub fn read_table(path: &Path) -> Result<Table> {
    let f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut reader = BufReader::new(f);
    let mut first = String::new();
    reader.read_line(&mut first)?;
    if first.is_empty() {
        return Ok(Table { rows: vec![] });
    }
    let sep = sniff_separator(first.trim_end());
    let mut rows = Vec::new();
    // 首行重新按分隔符切
    rows.push(split_line(first.trim_end(), sep));
    for line in reader.lines() {
        let line = line?;
        let line = line.trim_end_matches('\r');
        if line.is_empty() {
            continue;
        }
        rows.push(split_line(line, sep));
    }
    Ok(Table { rows })
}

fn split_line(line: &str, sep: char) -> Vec<String> {
    line.split(sep).map(|p| p.trim().to_string()).collect()
}

/// 便捷写 csv（行为对齐 pandas to_csv：逗号分隔、\n 结尾、无引号转义需求场景）
pub fn write_csv(path: &Path, header: &[&str], rows: &[Vec<String>]) -> Result<()> {
    use std::io::Write;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    if !header.is_empty() {
        writeln!(f, "{}", header.join(","))?;
    }
    for r in rows {
        writeln!(f, "{}", r.join(","))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sniff() {
        assert_eq!(sniff_separator("1,2,3,4"), ',');
        assert_eq!(sniff_separator("1\t2\t3"), '\t');
    }

    #[test]
    fn read_comma() {
        let dir = std::env::temp_dir().join("deepbsa_test_io");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.csv");
        std::fs::write(&p, "1,10,a,b,5,6\n2,20,c,d,7,8\n").unwrap();
        let t = read_table(&p).unwrap();
        assert_eq!(t.rows.len(), 2);
        assert_eq!(t.col_f64(1).unwrap(), vec![10.0, 20.0]);
        assert_eq!(t.col_str(0), vec!["1", "2"]);
    }
}
