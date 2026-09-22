//! 数据模拟，对齐 simulate_progress.py 的五步流程。
//! RNG 与 numpy 不同（原版本就不设种子，属随机过程，偏差 D5）。

use crate::npy;
use anyhow::{Context, Result};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::path::{Path, PathBuf};

#[derive(clap::Args)]
pub struct SimulateArgs {
    /// individual 数量
    #[arg(long = "i", required = true)]
    i: u32,
    /// pools 数量
    #[arg(long = "p", required = true)]
    p: u32,
    /// 高低池比例
    #[arg(long = "r", required = true)]
    r: f64,
    /// 效应位点数
    #[arg(long = "e", required = true)]
    e: u32,
    /// 输出目录
    #[arg(long = "s", required = true)]
    s: String,
}

/// 每条染色体单倍型数据（对齐 reduce_filtered_data/ch*.npy 的行 [chrom,pos,ref,mut]）
struct Chrom {
    rows: Vec<[f64; 4]>,
}

pub fn run(args: SimulateArgs) -> Result<()> {
    let cwd = std::env::current_dir()?;
    let sim_data_dir = cwd.join("Simulate_data");
    let save_path = PathBuf::from(&args.s);
    std::fs::create_dir_all(&save_path)?;

    // 载入模拟基础数据
    let integration = npy::read_path(&sim_data_dir.join("reduce_integration_data.npy"))?
        .as_f64()?;
    let mut chroms = Vec::new();
    let mut dir_list: Vec<PathBuf> = std::fs::read_dir(sim_data_dir.join("reduce_filtered_data"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|e| e == "npy").unwrap_or(false))
        .collect();
    dir_list.sort();
    for p in &dir_list {
        let a = npy::read_path(p)?;
        let flat = a.as_f64()?;
        let cols = a.cols();
        let rows: Vec<[f64; 4]> = (0..a.rows())
            .map(|r| [
                flat[r * cols],
                flat[r * cols + 1],
                flat[r * cols + 2],
                flat[r * cols + 3],
            ])
            .collect();
        chroms.push(Chrom { rows });
    }
    let total_len: usize = chroms.iter().map(|c| c.rows.len()).sum();

    println!("--Genetate Effective Points");
    let effective_points = finetune_ef(&integration, args.e)?;
    std::fs::create_dir_all(save_path.join("effective_points"))?;
    std::fs::write(
        save_path.join("effective_points").join(format!("{}_effective_points.json", args.e)),
        serde_json::to_string(&effective_points)?,
    )?;

    println!("--Generate Pairs Data");
    let generator_number = (f64::from(args.i) * 1.2).floor() as u32;
    let pairs_dir = save_path.join("saved_pairs_data");
    std::fs::create_dir_all(&pairs_dir)?;
    let normal5: Normal<f64> = Normal::new(5.0, 0.5).unwrap();
    let mut rng = rand::thread_rng();
    for count in 0..generator_number {
        let mut ref_result = Vec::with_capacity(total_len);
        let mut mut_result = Vec::with_capacity(total_len);
        for chrom in &chroms {
            let times = normal5.sample(&mut rng).round() as i64;
            let mut ref_h: Vec<f64> = chrom.rows.iter().map(|r| r[2]).collect();
            let mut mut_h: Vec<f64> = chrom.rows.iter().map(|r| r[3]).collect();
            for _ in 0..times {
                let point = rng.gen_range(0..chrom.rows.len());
                // recombine：交换两条单倍型的尾部
                let tail_ref = ref_h[point..].to_vec();
                let tail_mut = mut_h[point..].to_vec();
                ref_h[point..].copy_from_slice(&tail_mut);
                mut_h[point..].copy_from_slice(&tail_ref);
            }
            ref_result.extend(ref_h);
            mut_result.extend(mut_h);
        }
        // 自定义二进制：u64 n, f64*2n
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(pairs_dir.join(format!("{}.bin", count + 1)))?);
        f.write_all(&(ref_result.len() as u64).to_le_bytes())?;
        for v in &ref_result {
            f.write_all(&v.to_le_bytes())?;
        }
        for v in &mut_result {
            f.write_all(&v.to_le_bytes())?;
        }
        f.flush()?;
    }

    println!("Recombination");
    let max_i = args.i as usize;
    let gn = generator_number as usize;
    // 文件名从 1 开始（对齐原版 dir_list 的文件名字符串语义）
    let mut choosable: Vec<(usize, usize)> = (0..gn).flat_map(|i| [(i + 1, 0), (i + 1, 1)]).collect();
    let mut couples = Vec::with_capacity(max_i);
    for _ in 0..max_i {
        let a = rng.gen_range(0..choosable.len());
        let g1 = choosable.remove(a);
        let b = rng.gen_range(0..choosable.len());
        let g2 = choosable.remove(b);
        couples.push([g1, g2]);
    }

    println!("Calculate and Sort");
    // 随机抽 couples（不放回）
    let mut order: Vec<usize> = (0..couples.len()).collect();
    for k in (1..order.len()).rev() {
        let j = rng.gen_range(0..=k);
        order.swap(k, j);
    }
    let chosen = &order[..max_i.min(order.len())];
    let noise = Normal::new(0.0, 0.06).unwrap();
    let mut phenotype: Vec<([(usize, usize); 2], f64)> = Vec::with_capacity(chosen.len());
    for &ci in chosen {
        let couple = &couples[ci];
        let d1 = load_haplo(&pairs_dir, couple[0], total_len)?;
        let d2 = load_haplo(&pairs_dir, couple[1], total_len)?;
        let mut count = 0.0f64;
        for (idx, row) in &effective_points {
            let eff_ref = row[2];
            let eff_mut = row[3];
            let (a1, a2) = (d1[*idx], d2[*idx]);
            if a1 == eff_ref && a2 == eff_ref {
                count += 1.0 / args.e as f64;
            } else if a1 == eff_mut && a2 == eff_mut {
                // 无贡献
            } else {
                count += (1.0 / args.e as f64) / 2.0;
            }
        }
        count += noise.sample(&mut rng);
        count = count.clamp(0.0, 1.0);
        phenotype.push((*couple, count));
    }
    phenotype.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    std::fs::create_dir_all(save_path.join("cal_and_sort"))?;
    std::fs::write(
        save_path.join("cal_and_sort/sorted_phenotype.json"),
        serde_json::to_string(&phenotype)?,
    )?;

    println!("Generate Final Data");
    let dir_name = format!("{}-{}-{}", args.i, args.e, args.r);
    let iter_path = save_path.join(&dir_name).join(format!("{dir_name}--1"));
    std::fs::create_dir_all(&iter_path)?;
    divide_pools(&iter_path, args.i as usize, &save_path, args.p, args.r, &chroms)?;

    println!("done: {}", iter_path.display());
    Ok(())
}

fn load_haplo(dir: &Path, (file, strand): (usize, usize), _n: usize) -> Result<Vec<f64>> {
    use std::io::Read;
    // 文件结构: u64 n | f64 ref[n] | f64 mut[n]
    let mut f = std::fs::File::open(dir.join(format!("{file}.bin")))?;
    let mut lenbuf = [0u8; 8];
    f.read_exact(&mut lenbuf)?;
    let len = u64::from_le_bytes(lenbuf) as usize;
    let mut raw = vec![0u8; len * 2 * 8];
    f.read_exact(&mut raw)?;
    let vals: Vec<f64> = raw
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let start = strand * len;
    Ok(vals[start..start + len].to_vec())
}

/// 对齐 finetune_ef：随机抽 ep 个位点，位置间隔 >= min_dist（6e6，超时逐步放宽）
fn finetune_ef(integration: &[f64], ep_numbers: u32) -> Result<Vec<(usize, [f64; 4])>> {
    let cols = 4;
    let n_rows = integration.len() / cols;
    let row = |r: usize| -> [f64; 4] {
        [
            integration[r * cols],
            integration[r * cols + 1],
            integration[r * cols + 2],
            integration[r * cols + 3],
        ]
    };
    let mut rng = rand::thread_rng();
    let mut min_dist = 6e6f64;
    let start = std::time::Instant::now();
    let mut start_tmp = start;
    loop {
        if start_tmp.elapsed().as_secs() > 60 {
            min_dist *= 0.8;
            start_tmp = std::time::Instant::now();
            println!("reduce min distance");
        }
        let mut ef: Vec<(usize, [f64; 4])> = Vec::with_capacity(ep_numbers as usize);
        let mut distances: Vec<f64> = Vec::with_capacity(ep_numbers as usize);
        for _ in 0..ep_numbers {
            let choice = rng.gen_range(0..n_rows);
            ef.push((choice, row(choice)));
            distances.push(row(choice)[1]);
        }
        // 对齐原版：按 chrom 排序 ef；distances 升序
        ef.sort_by(|a, b| a.1[0].partial_cmp(&b.1[0]).unwrap());
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut ok = true;
        for (i, j) in (0..distances.len() - 1).zip(1..distances.len()) {
            if distances[j] - distances[i] < min_dist {
                ok = false;
                break;
            }
        }
        if ok || min_dist < 1.0 {
            return Ok(ef);
        }
    }
}

/// 对齐 divide_pools：按表型排序后切高/中/低池，输出 txt + csv
fn divide_pools(
    sp: &Path,
    pairs: usize,
    save_path: &Path,
    pool_num: u32,
    ratio: f64,
    chroms: &[Chrom],
) -> Result<()> {
    let sorted_raw = std::fs::read_to_string(save_path.join("cal_and_sort/sorted_phenotype.json"))?;
    let sorted_all: Vec<([(usize, usize); 2], f64)> =
        serde_json::from_str(&sorted_raw).context("parse phenotype json")?;
    let n = sorted_all.len();
    let mut idx: Vec<usize> = (0..n).collect();
    // 随机抽 pairs 个（不放回）
    let mut rng = rand::thread_rng();
    for k in (1..n).rev() {
        let j = rng.gen_range(0..=k);
        idx.swap(k, j);
    }
    idx.truncate(pairs);
    let mut chosen: Vec<&([(usize, usize); 2], f64)> = idx.iter().map(|&i| &sorted_all[i]).collect();
    chosen.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let save_dir = sp.join("divide_pools");
    std::fs::create_dir_all(&save_dir)?;
    let pool_size = pairs / pool_num as usize;
    let n_pool_members = (pairs as f64 * ratio) as usize;

    let mut summary: Vec<Vec<f64>> = Vec::with_capacity(pool_num as usize);
    for index in 0..pool_num as usize {
        let pool_start = pool_size * index;
        let pool_name: Vec<&([(usize, usize); 2], f64)> = if index == 0 {
            chosen[..n_pool_members].to_vec()
        } else if index == pool_num as usize - 1 {
            chosen[chosen.len() - n_pool_members..].to_vec()
        } else {
            let s = (pool_size - n_pool_members) / 2 + pool_start;
            chosen[s..s + n_pool_members].to_vec()
        };
        // 池内单倍型逐元素求和
        let mut pool_data = vec![0.0f64; chroms.iter().map(|c| c.rows.len()).sum::<usize>()];
        for (name, _) in &pool_name {
            let h1 = load_haplo(&save_path.join("saved_pairs_data"), name[0], pool_data.len())?;
            let h2 = load_haplo(&save_path.join("saved_pairs_data"), name[1], pool_data.len())?;
            for k in 0..pool_data.len() {
                pool_data[k] += h1[k] + h2[k];
            }
        }
        // 对齐原版：除以 int(pairs/pool_num)*2（不按实际池大小）
        let denom = (pool_size * 2) as f64;
        summary.push(pool_data.into_iter().map(|v| v / denom).collect());
    }

    // txt + csv（对齐原版格式）
    let integration = npy::read_path(
        &std::env::current_dir()?.join("Simulate_data/reduce_integration_data.npy"),
    )?
    .as_f64()?;
    let mut txt = String::new();
    let mut csv_rows: Vec<Vec<String>> = Vec::new();
    for r in 0..integration.len() / 4 {
        let chrom = integration[r * 4] as i64;
        let pos = integration[r * 4 + 1];
        let mut line = format!("{chrom}\t{pos}\tI\tI");
        let mut csv_row = vec![chrom.to_string(), format!("{pos}"), "I".into(), "I".into()];
        for pool in &summary {
            let k = pool[r];
            line += &format!("\t{:.3},{:.3}", k, 1.0 - k);
            csv_row.push(format!("{k}"));
            csv_row.push(format!("{}", 1.0 - k));
        }
        txt += &line;
        txt.push('\n');
        csv_rows.push(csv_row);
    }
    std::fs::write(save_dir.join("simulate_data.txt"), txt)?;
    let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_millis();
    crate::io::write_csv(&save_dir.join(format!("simulate_data_{ts}.csv")), &[], &csv_rows)?;
    Ok(())
}
