//! 最小 npy (v1.0) 读写，仅覆盖本项目用到的 dtype/维度组合。
//! 模拟数据读取需要：f64/f32/i64/i32 的 1D/2D 数组。

use anyhow::{bail, Context, Result};
use std::io::{Read, Write};

#[derive(Debug, Clone, PartialEq)]
pub enum NpyData {
    F64(Vec<f64>),
    F32(Vec<f32>),
    I64(Vec<i64>),
    I32(Vec<i32>),
}

#[derive(Debug, Clone)]
pub struct NpyArray {
    #[allow(dead_code)]
    pub descr: String,
    pub shape: Vec<usize>,
    pub data: NpyData,
}

impl NpyArray {
    /// 展平后的 f64 视图（f32/i64/i32 会提升转换）
    pub fn as_f64(&self) -> Result<Vec<f64>> {
        Ok(match &self.data {
            NpyData::F64(v) => v.clone(),
            NpyData::F32(v) => v.iter().map(|&x| x as f64).collect(),
            NpyData::I64(v) => v.iter().map(|&x| x as f64).collect(),
            NpyData::I32(v) => v.iter().map(|&x| x as f64).collect(),
        })
    }

    pub fn rows(&self) -> usize {
        if self.shape.is_empty() {
            0
        } else {
            self.shape[0]
        }
    }

    pub fn cols(&self) -> usize {
        match self.shape.len() {
            0 => 0,
            1 => 1,
            _ => self.shape[1],
        }
    }

    /// 取第 r 行（按 C 序）
    #[allow(dead_code)]
    pub fn row(&self, r: usize) -> Result<Vec<f64>> {
        let flat = self.as_f64()?;
        let c = self.cols().max(1);
        let start = r * c;
        Ok(flat[start..start + c].to_vec())
    }
}

fn parse_header(header: &str) -> Result<(String, Vec<usize>)> {
    // python dict 字面量，手写窄解析即可
    let descr = header
        .split("'descr':")
        .nth(1)
        .and_then(|s| s.split(',').next())
        .map(|s| s.trim().trim_matches(|c| c == '\'' || c == '"').to_string())
        .context("npy header missing descr")?;

    let shape_part = header
        .split("'shape':")
        .nth(1)
        .and_then(|s| s.split('}').next())
        .context("npy header missing shape")?;
    let shape: Vec<usize> = shape_part
        .trim()
        .trim_start_matches('(')
        .trim_end_matches("),")
        .trim_end_matches(')')
        .trim()
        .trim_end_matches(',')
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse().context("bad shape dim"))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((descr, shape))
}

pub fn read<R: Read>(mut r: R) -> Result<NpyArray> {
    let mut magic_ver = [0u8; 8];
    r.read_exact(&mut magic_ver)?;
    if &magic_ver[..6] != b"\x93NUMPY" {
        bail!("not a npy file");
    }
    let (major, header_len) = match magic_ver[6] {
        1 => {
            let mut l = [0u8; 2];
            r.read_exact(&mut l)?;
            (1u8, u16::from_le_bytes(l) as usize)
        }
        2 | 3 => {
            let mut l = [0u8; 4];
            r.read_exact(&mut l)?;
            (magic_ver[6], u32::from_le_bytes(l) as usize)
        }
        v => bail!("unsupported npy major version {v}"),
    };
    let mut header_buf = vec![0u8; header_len];
    r.read_exact(&mut header_buf)?;
    let header = String::from_utf8(header_buf)?;
    let (descr, shape) = parse_header(&header).with_context(|| format!("header: {header}"))?;

    let n: usize = shape.iter().product::<usize>().max(if shape.is_empty() { 1 } else { 0 });
    let data = match descr.as_str() {
        "<f8" | "|f8" | "float64" => {
            let mut raw = vec![0u8; n * 8];
            r.read_exact(&mut raw)?;
            NpyData::F64(raw.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect())
        }
        "<f4" | "|f4" | "float32" => {
            let mut raw = vec![0u8; n * 4];
            r.read_exact(&mut raw)?;
            NpyData::F32(raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
        }
        "<i8" | "|i8" | "int64" => {
            let mut raw = vec![0u8; n * 8];
            r.read_exact(&mut raw)?;
            NpyData::I64(raw.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect())
        }
        "<i4" | "|i4" | "int32" => {
            let mut raw = vec![0u8; n * 4];
            r.read_exact(&mut raw)?;
            NpyData::I32(raw.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect())
        }
        "|O" | "O" | "object" => bail!("object(pickle) npy 不支持读取：原版 object 数组已改为自定义二进制格式，见设计文档 D2"),
        d => bail!("unsupported dtype {d}"),
    };
    let _ = major;
    Ok(NpyArray { descr, shape, data })
}

pub fn read_path(path: &std::path::Path) -> Result<NpyArray> {
    let f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut br = std::io::BufReader::new(f);
    read(&mut br)
}

fn header_for(descr: &str, shape: &[usize]) -> Vec<u8> {
    let shape_str = match shape.len() {
        0 => "(), ".to_string(),
        1 => format!("({},)", shape[0]),
        _ => format!("({},)", shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(",")),
    };
    let dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_str}, }}");
    // 头部按 64 字节对齐（v1.0 规范）
    let dict_len = dict.len();
    let pad = 64 - ((10 + dict_len) % 64); // 10 = magic(6)+ver(2)+len(2)
    let total = dict_len + pad;
    let mut out = Vec::with_capacity(10 + total);
    out.extend_from_slice(b"\x93NUMPY");
    out.extend_from_slice(&[1u8, 0u8]);
    out.extend_from_slice(&(total as u16).to_le_bytes());
    out.extend_from_slice(dict.as_bytes());
    out.extend(std::iter::repeat(b' ').take(pad));
    out
}

/// 写 f64 一维数组
pub fn write_f64_1d<W: Write>(w: &mut W, data: &[f64]) -> Result<()> {
    w.write_all(&header_for("<f8", &[data.len()]))?;
    for v in data {
        w.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

/// 写 f64 二维数组（C 序）
#[allow(dead_code)]
pub fn write_f64_2d<W: Write>(w: &mut W, rows: usize, cols: usize, data: &[f64]) -> Result<()> {
    assert_eq!(rows * cols, data.len());
    w.write_all(&header_for("<f8", &[rows, cols]))?;
    for v in data {
        w.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

/// 写 numpy **object 数组**（dtype '|O'，pickle 协议 3）——
/// 与原版 np.save(每染色体不等长列表) 的格式等价，np.load(allow_pickle=True)
/// 后 len(arr)==染色体数、len(arr[i])==该染色体数据点数。
/// 这是 biopytools deepbsa merge npy 快速路径所要求的真实格式。
///
/// pickle 字节模板取自 numpy 对 1 行空 object 数组的真实序列化输出
/// （scripts/gen_pickle_template.py），行块中的 memo 引用（h\x00/h\x01/h\x03/h\x07）
/// 全部指向 prefix 的全局对象，跨行复用安全；BINPUT 槽位号重复使用合法。
pub fn write_object_ragged<W: Write>(w: &mut W, rows: &[Vec<f64>]) -> Result<()> {
    // ---- 头部（与 numpy 规范一致：空格填充 + 换行终止）----
    let shape_str = format!("({},)", rows.len());
    let dict = format!("{{'descr': '|O', 'fortran_order': False, 'shape': {shape_str}, }}");
    let dict_len = dict.len();
    let total = dict_len + (64 - ((10 + dict_len) % 64));
    w.write_all(b"\x93NUMPY\x01\x00")?;
    w.write_all(&(total as u16).to_le_bytes())?;
    w.write_all(dict.as_bytes())?;
    let pad = total - dict_len;
    let mut pad_buf = vec![b' '; pad.saturating_sub(1)];
    pad_buf.push(b'\n');
    w.write_all(&pad_buf)?;

    // ---- pickle ----
    let mut p: Vec<u8> = Vec::new();
    p.extend_from_slice(b"\x80\x03");
    // 外层: _reconstruct(ndarray, (0,), b'b')
    p.extend_from_slice(b"cnumpy.core.multiarray\n_reconstruct\nq\x00");
    p.extend_from_slice(b"cnumpy\nndarray\nq\x01");
    p.extend_from_slice(b"K\x00\x85q\x02C\x01bq\x03\x87q\x04Rq\x05");
    // setstate 参数元组开始
    p.push(b'(');
    // shape (1, n)
    p.extend_from_slice(b"K\x01");
    encode_int(&mut p, rows.len() as u64);
    p.extend_from_slice(b"\x85q\x06");
    // dtype('O8')
    p.extend_from_slice(b"cnumpy\ndtype\nq\x07X\x02\x00\x00\x00O8q\x08\x89\x88\x87q\tRq\n");
    p.extend_from_slice(b"(K\x03X\x01\x00\x00\x00|q\x0bNNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK?tq\x0cb");
    // fortran_order = False
    p.push(b'\x89');
    // 数据 list（各行随后用 APPEND 逐个挂入）
    p.extend_from_slice(b"]q\r");
    for row in rows {
        // 行对象: _reconstruct(ndarray, (0,), b'b')，setstate=(shape(1,len), dtype('f8'), False, data)
        p.extend_from_slice(b"h\x00h\x01K\x00\x85q\x0eh\x03\x87q\x0fRq\x10(K\x01");
        encode_int(&mut p, row.len() as u64);
        p.extend_from_slice(b"\x85q\x11h\x07X\x02\x00\x00\x00f8q\x12\x89\x88\x87q\x13Rq\x14");
        p.extend_from_slice(b"(K\x03X\x01\x00\x00\x00<q\x15NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tq\x16b\x89");
        // 行数据: 字节数<=255 用 SHORT_BINBYTES('C')，否则 BINBYTES('B', u32)
        // 注意：'T' 是废弃的 BINSTRING（会按字符串解码），不可用
        let bytes_len = row.len() * 8;
        if bytes_len <= 255 {
            p.push(b'C');
            p.push(bytes_len as u8);
        } else {
            p.push(b'B');
            p.extend_from_slice(&(bytes_len as u32).to_le_bytes());
        }
        for &v in row {
            p.extend_from_slice(&v.to_le_bytes());
        }
        p.extend_from_slice(b"q\x17tq\x18b");
        // APPEND: list(栈顶第二个).append(row)
        p.push(b'a');
    }
    // 外层 setstate 元组收口 + BUILD + STOP
    p.extend_from_slice(b"tq\x19b.");
    w.write_all(&p)?;
    Ok(())
}

/// pickle 整数编码：<256 用 BININT1(K)，否则 LONG1(\x8a + u8 字节数 + LE)
fn encode_int(p: &mut Vec<u8>, v: u64) {
    if v < 256 {
        p.push(b'K');
        p.push(v as u8);
    } else if v <= u32::MAX as u64 {
        p.push(0x8a); // LONG1
        p.push(4);
        p.extend_from_slice(&(v as u32).to_le_bytes());
    } else {
        p.push(0x8a); // LONG1
        p.push(8);
        p.extend_from_slice(&v.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_1d() {
        let mut buf = Vec::new();
        write_f64_1d(&mut buf, &[1.0, -2.5, 3.25]).unwrap();
        let a = read(&mut &buf[..]).unwrap();
        assert_eq!(a.shape, vec![3]);
        assert_eq!(a.as_f64().unwrap(), vec![1.0, -2.5, 3.25]);
    }

    #[test]
    fn roundtrip_2d() {
        let mut buf = Vec::new();
        write_f64_2d(&mut buf, 2, 3, &[1., 2., 3., 4., 5., 6.]).unwrap();
        let a = read(&mut &buf[..]).unwrap();
        assert_eq!(a.shape, vec![2, 3]);
        assert_eq!(a.row(1).unwrap(), vec![4., 5., 6.]);
    }
}

#[cfg(test)]
mod object_npy_tests {
    use super::*;

    #[test]
    fn object_npy_structure() {
        let mut buf = Vec::new();
        let mut big = vec![0.5f64; 100]; // 800 字节 -> 走 BINBYTES('B') 路径
        big[0] = 1.0;
        write_object_ragged(&mut buf, &[vec![1.0, 2.0], vec![], vec![3.0], big]).unwrap();
        // 头部
        assert_eq!(&buf[..6], b"\x93NUMPY");
        assert_eq!(buf[6], 1); // v1.0
        let hlen = u16::from_le_bytes([buf[8], buf[9]]) as usize;
        let header = String::from_utf8(buf[10..10 + hlen].to_vec()).unwrap();
        assert!(header.contains("'descr': '|O'"), "{header}");
        assert!(header.contains("'shape': (4,)"), "{header}");
        assert_eq!(*buf.last().unwrap(), b'.'); // pickle STOP
        // pickle 骨架: 协议 3 + 数据行使用 BINBYTES('B')
        let pickle = &buf[10 + hlen..];
        assert_eq!(pickle[0], 0x80);
        assert_eq!(pickle[1], 3);
        // 数据行必须用 BINBYTES('B')，不得出现废弃的 BINSTRING('T')
        assert!(pickle.windows(2).any(|w| w == b"\x89B"), "must use BINBYTES");
        assert!(pickle.windows(2).any(|w| w == b"\x89C"), "must use SHORT_BINBYTES");
        assert!(!pickle.windows(2).any(|w| w == b"\x89T"), "must not use BINSTRING");
    }
}
