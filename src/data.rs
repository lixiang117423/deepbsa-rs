//! 核心数据结构：按染色体组织的池数据，以及与原版 .npy(object) 缓存
//! 对应的自定义二进制缓存（设计文档 D2）。

use anyhow::{bail, Context, Result};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// 一条染色体的预处理结果：n_snps × n_pools 矩阵 + 位置
#[derive(Debug, Clone, Default)]
pub struct ChrData {
    pub ref_a: Vec<Vec<f64>>,  // A(ref) 深度矩阵
    pub mut_a: Vec<Vec<f64>>,  // a(alt) 深度矩阵
    pub freq: Vec<Vec<f64>>,   // A/(A+a)
    pub pos: Vec<f64>,
}

impl ChrData {
    pub fn len(&self) -> usize {
        self.pos.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pos.is_empty()
    }

    pub fn n_pools(&self) -> usize {
        self.ref_a.first().map(|r| r.len()).unwrap_or(0)
    }
}

#[derive(Debug, Clone, Default)]
pub struct Dataset {
    /// 染色体名（首次出现顺序，对齐原版 Counter 语义）
    pub chrome_set: Vec<String>,
    pub chrs: Vec<ChrData>,
}

impl Dataset {
    pub fn from_table(table: &crate::io::Table) -> Result<Dataset> {
        if table.rows.is_empty() {
            bail!("input table is empty");
        }
        let n_cols = table.n_cols();
        if n_cols < 6 || (n_cols - 4) % 2 != 0 {
            bail!(
                "expect columns: chrom,pos,ref,alt,+2*n_pools, got {n_cols}"
            );
        }
        let mut ds = Dataset::default();
        let mut idx: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for row in &table.rows {
            let chrom = row[0].clone();
            let chr_idx = match idx.get(&chrom) {
                Some(&i) => i,
                None => {
                    let i = ds.chrome_set.len();
                    ds.chrome_set.push(chrom.clone());
                    ds.chrs.push(ChrData::default());
                    idx.insert(chrom, i);
                    i
                }
            };
            let pos: f64 = row[1].parse().context("parse position")?;
            // 对齐原版 row[4::2] / row[5::2]
            let mut ap = Vec::with_capacity((n_cols - 4) / 2);
            let mut am = Vec::with_capacity((n_cols - 4) / 2);
            let mut k = 4;
            while k + 1 < n_cols {
                ap.push(row[k].parse::<f64>().context("parse A depth")?);
                am.push(row[k + 1].parse::<f64>().context("parse a depth")?);
                k += 2;
            }
            let chr = &mut ds.chrs[chr_idx];
            chr.pos.push(pos);
            chr.ref_a.push(ap.clone());
            chr.mut_a.push(am);
            chr.freq.push(vec![0.0; ap.len()]);
        }
        Ok(ds)
    }

    pub fn n_pools(&self) -> usize {
        self.chrs.iter().map(|c| c.n_pools()).max().unwrap_or(0)
    }

}

// ---------------- 自定义二进制缓存（D2） ----------------
// magic "DBC1" | u32 n_chr
// 每条染色体: u32 n_snps, u32 n_pools,
//   f64*n_snps*n_pools (ref) , f64*(..) (mut), f64*(..) (freq), f64*n_snps (pos)
// u32 chrome_set_len; 每项: u32 len, utf8 bytes

const CACHE_MAGIC: &[u8; 4] = b"DBC1";

fn write_f64s<W: Write>(w: &mut W, v: &[f64]) -> Result<()> {
    let mut buf = Vec::with_capacity(v.len() * 8);
    for x in v {
        buf.extend_from_slice(&x.to_le_bytes());
    }
    w.write_all(&buf)?;
    Ok(())
}

fn read_exact_f64s<R: Read>(r: &mut R, n: usize) -> Result<Vec<f64>> {
    let mut raw = vec![0u8; n * 8];
    r.read_exact(&mut raw)?;
    Ok(raw.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect())
}

pub fn save_cache(path: &Path, ds: &Dataset) -> Result<()> {
    let mut w = BufWriter::new(std::fs::File::create(path)?);
    w.write_all(CACHE_MAGIC)?;
    w.write_all(&(ds.chrs.len() as u32).to_le_bytes())?;
    for chr in &ds.chrs {
        let pools = chr.n_pools();
        w.write_all(&(chr.len() as u32).to_le_bytes())?;
        w.write_all(&(pools as u32).to_le_bytes())?;
        let mut flat = Vec::with_capacity(chr.len() * pools);
        for row in &chr.ref_a {
            flat.extend_from_slice(row);
        }
        write_f64s(&mut w, &flat)?;
        flat.clear();
        for row in &chr.mut_a {
            flat.extend_from_slice(row);
        }
        write_f64s(&mut w, &flat)?;
        flat.clear();
        for row in &chr.freq {
            flat.extend_from_slice(row);
        }
        write_f64s(&mut w, &flat)?;
        write_f64s(&mut w, &chr.pos)?;
    }
    w.write_all(&(ds.chrome_set.len() as u32).to_le_bytes())?;
    for name in &ds.chrome_set {
        w.write_all(&(name.len() as u32).to_le_bytes())?;
        w.write_all(name.as_bytes())?;
    }
    w.flush()?;
    Ok(())
}

pub fn load_cache(path: &Path) -> Result<Dataset> {
    let mut r = BufReader::new(std::fs::File::open(path)?);
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != CACHE_MAGIC {
        bail!("bad cache magic in {}", path.display());
    }
    let mut u32buf = [0u8; 4];
    r.read_exact(&mut u32buf)?;
    let n_chr = u32::from_le_bytes(u32buf) as usize;
    let mut chrs = Vec::with_capacity(n_chr);
    for _ in 0..n_chr {
        r.read_exact(&mut u32buf)?;
        let n_snps = u32::from_le_bytes(u32buf) as usize;
        r.read_exact(&mut u32buf)?;
        let pools = u32::from_le_bytes(u32buf) as usize;
        let n = n_snps * pools;
        let ref_flat = read_exact_f64s(&mut r, n)?;
        let mut_flat = read_exact_f64s(&mut r, n)?;
        let freq_flat = read_exact_f64s(&mut r, n)?;
        let pos = read_exact_f64s(&mut r, n_snps)?;
        let mut chr = ChrData {
            pos,
            ..Default::default()
        };
        for s in 0..n_snps {
            chr.ref_a.push(ref_flat[s * pools..(s + 1) * pools].to_vec());
            chr.mut_a.push(mut_flat[s * pools..(s + 1) * pools].to_vec());
            chr.freq.push(freq_flat[s * pools..(s + 1) * pools].to_vec());
        }
        chrs.push(chr);
    }
    r.read_exact(&mut u32buf)?;
    let n_names = u32::from_le_bytes(u32buf) as usize;
    let mut chrome_set = Vec::with_capacity(n_names);
    for _ in 0..n_names {
        r.read_exact(&mut u32buf)?;
        let len = u32::from_le_bytes(u32buf) as usize;
        let mut name = vec![0u8; len];
        r.read_exact(&mut name)?;
        chrome_set.push(String::from_utf8(name)?);
    }
    Ok(Dataset { chrome_set, chrs })
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::read_table;

    fn make_csv() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("deepbsa_test_data");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("ds.csv");
        std::fs::write(&p, "1,100,A,G,10,5,8,6\n1,200,C,T,12,4,9,5\n2,300,A,G,20,8,15,7\n").unwrap();
        p
    }

    #[test]
    fn parse_and_cache_roundtrip() {
        let p = make_csv();
        let t = read_table(&p).unwrap();
        let ds = Dataset::from_table(&t).unwrap();
        assert_eq!(ds.chrome_set, vec!["1", "2"]);
        assert_eq!(ds.chrs[0].len(), 2);
        assert_eq!(ds.chrs[1].len(), 1);
        assert_eq!(ds.n_pools(), 2);

        let cache = std::env::temp_dir().join("deepbsa_test_data/c.bin");
        save_cache(&cache, &ds).unwrap();
        let ds2 = load_cache(&cache).unwrap();
        assert_eq!(ds2.chrome_set, ds.chrome_set);
        assert_eq!(ds2.chrs[0].ref_a, ds.chrs[0].ref_a);
        assert_eq!(ds2.chrs[1].pos, ds.chrs[1].pos);
    }
}
