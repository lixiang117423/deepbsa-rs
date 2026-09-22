//! DL 推理：加载 convert_weights.py 产出的自定义权重文件，
//! 手写 Conv1D 自编码器前向传播（方案 A，无任何外部推理引擎）。
//!
//! 权重格式见 scripts/convert_weights.py 顶部注释。
//! 'same' 填充遵循 TF 约定：pad_left = (k-1)/2，pad_right = k-1-pad_left。

use anyhow::{bail, Context, Result};
use ndarray::Array3;
use rayon::prelude::*;
use std::io::Read;
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 4] = b"DBW1";

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Act {
    None,
    Relu,
    Sigmoid,
}

#[derive(Debug)]
pub enum Layer {
    Input,
    Conv1D {
        kernel: usize,
        pad_same: bool,
        act: Act,
        weight: Array3<f32>, // (k, c_in, c_out)
        bias: Option<Vec<f32>>,
    },
    MaxPool1D { pool: usize, stride: usize },
    UpSample1D { size: usize },
    Add,
    Dropout,
}

#[derive(Debug)]
pub struct DLModel {
    layers: Vec<(Layer, Vec<usize>)>, // (层, 入边)
}

pub fn weights_dir(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(p) = explicit {
        if p.is_dir() {
            return Ok(p.to_path_buf());
        }
        bail!("weights dir not found: {}", p.display());
    }
    if let Ok(env) = std::env::var("DEEPBSA_WEIGHTS") {
        let p = PathBuf::from(env);
        if p.is_dir() {
            return Ok(p);
        }
    }
    // 可执行文件同级
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let p = dir.join("weights");
            if p.is_dir() {
                return Ok(p);
            }
            // 再向上一级（target/release -> 项目根）
            if let Some(up) = dir.parent() {
                let p = up.join("weights");
                if p.is_dir() {
                    return Ok(p);
                }
            }
        }
    }
    let p = PathBuf::from("weights");
    if p.is_dir() {
        return Ok(p);
    }
    bail!("weights directory not found (use --weights or DEEPBSA_WEIGHTS)")
}

/// 二进制游标
struct Cur<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Cur<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.p + n > self.b.len() {
            bail!("weights file truncated");
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn u32le(&mut self) -> Result<u32> {
        let s = self.take(4)?;
        Ok(u32::from_le_bytes(s.try_into().unwrap()))
    }
}

fn read_f32s(cur: &mut Cur, n: usize) -> Result<Vec<f32>> {
    let bytes = cur.take(n * 4)?;
    Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}

fn read_array3(cur: &mut Cur) -> Result<Array3<f32>> {
    let ndim = cur.u32le()?;
    if ndim != 3 {
        bail!("expected 3d array, got ndim={ndim}");
    }
    let d0 = cur.u32le()? as usize;
    let d1 = cur.u32le()? as usize;
    let d2 = cur.u32le()? as usize;
    let data = read_f32s(cur, d0 * d1 * d2)?;
    Ok(Array3::from_shape_vec((d0, d1, d2), data).unwrap())
}

impl DLModel {
    pub fn load(path: &Path) -> Result<DLModel> {
        let mut raw = Vec::new();
        std::fs::File::open(path)
            .with_context(|| format!("open {}", path.display()))?
            .read_to_end(&mut raw)?;
        let mut cur = Cur { b: &raw, p: 0 };
        if cur.take(4)? != MAGIC {
            bail!("bad weights magic");
        }
        let n_layers = cur.u32le()? as usize;
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            let op = cur.u32le()?;
            let n_in = cur.u32le()? as usize;
            let mut inputs = Vec::with_capacity(n_in);
            for _ in 0..n_in {
                inputs.push(cur.u32le()? as usize);
            }
            let layer = match op {
                4 => Layer::Input,
                5 => Layer::Dropout,
                3 => Layer::Add,
                2 => {
                    let size = cur.u32le()? as usize;
                    Layer::UpSample1D { size }
                }
                1 => {
                    let pool = cur.u32le()? as usize;
                    let stride = cur.u32le()? as usize;
                    Layer::MaxPool1D { pool, stride }
                }
                0 => {
                    let kernel = cur.u32le()? as usize;
                    let pad_code = cur.u32le()?;
                    let act = match cur.u32le()? {
                        0 => Act::None,
                        1 => Act::Relu,
                        2 => Act::Sigmoid,
                        v => bail!("unknown act code {v}"),
                    };
                    let n_arrays = cur.u32le()? as usize;
                    if n_arrays < 1 {
                        bail!("conv without kernel");
                    }
                    let weight = read_array3(&mut cur)?;
                    let bias = if n_arrays >= 2 {
                        let ndim = cur.u32le()? as usize;
                        if ndim != 1 {
                            bail!("bias must be 1d");
                        }
                        let n = cur.u32le()? as usize;
                        Some(read_f32s(&mut cur, n)?)
                    } else {
                        None
                    };
                    Layer::Conv1D {
                        kernel,
                        pad_same: pad_code == 0,
                        act,
                        weight,
                        bias,
                    }
                }
                v => bail!("unknown op code {v}"),
            };
            layers.push((layer, inputs));
        }
        Ok(DLModel { layers })
    }

    pub fn load_for_pools(dir: &Path, pools: usize) -> Result<DLModel> {
        if !(2..=10).contains(&pools) {
            bail!("DL method supports 2..=10 pools, got {pools}");
        }
        Self::load(&dir.join(format!("{pools}pool.bin")))
    }

    /// x: (batch, L, c_in) -> (batch, L, c_out)
    pub fn forward(&self, x: &Array3<f32>) -> Array3<f32> {
        let mut vals: Vec<Option<Array3<f32>>> = vec![None; self.layers.len()];
        for (li, (layer, inputs)) in self.layers.iter().enumerate() {
            let out = match layer {
                Layer::Input => x.clone(),
                Layer::Dropout => vals[inputs[0]].as_ref().unwrap().clone(),
                Layer::Add => {
                    let a = vals[inputs[0]].as_ref().unwrap();
                    let b = vals[inputs[1]].as_ref().unwrap();
                    a + b
                }
                Layer::UpSample1D { size } => {
                    let v = vals[inputs[0]].as_ref().unwrap();
                    upsample1d(v, *size)
                }
                Layer::MaxPool1D { pool, stride } => {
                    let v = vals[inputs[0]].as_ref().unwrap();
                    maxpool1d(v, *pool, *stride)
                }
                Layer::Conv1D {
                    kernel,
                    pad_same,
                    act,
                    weight,
                    bias,
                } => {
                    let v = vals[inputs[0]].as_ref().unwrap();
                    conv1d(v, *kernel, *pad_same, *act, weight, bias.as_deref())
                }
            };
            vals[li] = Some(out);
        }
        vals.last().unwrap().as_ref().unwrap().clone()
    }
}

/// 对齐原版 get_data 的 DL 分支：
/// 零填充到 64 的倍数（len%64==0 时额外补一个全零窗，输出会被截掉），
/// 非重叠切窗推理后展平截取前 n 个值。
pub fn dl_statistic(model: &DLModel, freq: &Vec<Vec<f64>>) -> Vec<f64> {
    let n = freq.len();
    let pools = freq[0].len();
    let a = n % 64;
    let pad = 64 - a;
    let total = n + pad;
    let batch = total / 64;
    let mut x = Array3::<f32>::zeros((batch, 64, pools));
    for (i, row) in freq.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            x[[i / 64, i % 64, j]] = v as f32;
        }
    }
    let y = model.forward(&x);
    y.as_slice()
        .unwrap()
        .iter()
        .take(n)
        .map(|&v| v as f64)
        .collect()
}

/// TF 'SAME'（stride=1）：pad_left = (k-1)/2, pad_right = k-1-pad_left
fn conv1d(
    x: &Array3<f32>,
    k: usize,
    pad_same: bool,
    act: Act,
    weight: &Array3<f32>,
    bias: Option<&[f32]>,
) -> Array3<f32> {
    let (b, l, c_in) = (x.shape()[0], x.shape()[1], x.shape()[2]);
    let c_out = weight.shape()[2];
    let (pad_l, pad_r) = if pad_same {
        let total = k - 1;
        (total / 2, total - total / 2)
    } else {
        (0, 0)
    };
    let l_out = l + pad_l + pad_r + 1 - k;
    let mut out = Array3::<f32>::zeros((b, l_out, c_out));
    let w = weight.as_slice().unwrap();
    let x_view = x.view();

    // 按 batch 并行；权重布局 (k, c_in, c_out) 展平后对内层累加连续
    out.axis_iter_mut(ndarray::Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(bb, mut batch)| {
            for t in 0..l_out {
                let mut acc = vec![0f32; c_out];
                for dt in 0..k {
                    let src = t as isize + dt as isize - pad_l as isize;
                    if src < 0 || src as usize >= l {
                        continue;
                    }
                    let src = src as usize;
                    for ci in 0..c_in {
                        let xv = x_view[[bb, src, ci]];
                        if xv == 0.0 {
                            continue;
                        }
                        let base = (dt * c_in + ci) * c_out;
                        for co in 0..c_out {
                            acc[co] += xv * w[base + co];
                        }
                    }
                }
                if let Some(bs) = bias {
                    for co in 0..c_out {
                        acc[co] += bs[co];
                    }
                }
                match act {
                    Act::None => {}
                    Act::Relu => {
                        for v in acc.iter_mut() {
                            if *v < 0.0 {
                                *v = 0.0;
                            }
                        }
                    }
                    Act::Sigmoid => {
                        for v in acc.iter_mut() {
                            *v = 1.0 / (1.0 + (-*v as f64).exp() as f32);
                        }
                    }
                }
                for co in 0..c_out {
                    batch[[t, co]] = acc[co];
                }
            }
        });
    out
}

fn maxpool1d(x: &Array3<f32>, pool: usize, stride: usize) -> Array3<f32> {
    let (b, l, c) = (x.shape()[0], x.shape()[1], x.shape()[2]);
    let l_out = (l - pool) / stride + 1;
    let mut out = Array3::<f32>::zeros((b, l_out, c));
    for bb in 0..b {
        for t in 0..l_out {
            for cc in 0..c {
                let mut m = f32::NEG_INFINITY;
                for p in 0..pool {
                    let idx = t * stride + p;
                    let v = x[[bb, idx, cc]];
                    if v > m {
                        m = v;
                    }
                }
                out[[bb, t, cc]] = m;
            }
        }
    }
    out
}

fn upsample1d(x: &Array3<f32>, size: usize) -> Array3<f32> {
    let (b, l, c) = (x.shape()[0], x.shape()[1], x.shape()[2]);
    let mut out = Array3::<f32>::zeros((b, l * size, c));
    for bb in 0..b {
        for t in 0..l {
            for s in 0..size {
                for cc in 0..c {
                    out[[bb, t * size + s, cc]] = x[[bb, t, cc]];
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 对拍 TF 黄金标准（fixtures 由 scripts/verify_tf_vs_numpy.py 生成）。
    /// 测试输入: (189, 64, 2) 窗口；期望输出: TF model.predict 结果。
    #[test]
    fn dl_forward_matches_tf() {
        let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let fx = base.join("tests_fixtures");
        if !fx.join("dl_test_input.npy").exists() {
            eprintln!("fixtures missing, skip");
            return;
        }
        let weights = base.join("weights");
        let model = DLModel::load_for_pools(&weights, 2).unwrap();
        let x = crate::npy::read_path(&fx.join("dl_test_input.npy")).unwrap();
        let expected = crate::npy::read_path(&fx.join("dl_test_output.npy")).unwrap();

        let (b, l, c) = (x.shape[0], x.shape[1], x.shape[2]);
        let xf = x.as_f64().unwrap();
        let mut input = Array3::<f32>::zeros((b, l, c));
        for i in 0..b {
            for j in 0..l {
                for k in 0..c {
                    input[[i, j, k]] = xf[(i * l + j) * c + k] as f32;
                }
            }
        }
        let out = model.forward(&input);
        let ef = expected.as_f64().unwrap();
        let max_err = out
            .as_slice()
            .unwrap()
            .iter()
            .zip(ef.iter())
            .map(|(&a, &b)| (a as f64 - b).abs())
            .fold(0.0f64, f64::max);
        assert!(max_err < 1e-5, "max err {max_err}");
        println!("DL forward max err vs TF = {max_err:.3e}");
    }
}
