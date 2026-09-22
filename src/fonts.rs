//! 内嵌字体：DejaVu Sans（自由许可，可再分发）。
//! 通过 plotters 的 ab_glyph 后端注册，避免 fontconfig/freetype 等 C 依赖，
//! 使 Linux 下可静态链接（musl）且运行环境无需安装任何字体。

use plotters::style::register_font;

const SANS_REGULAR: &[u8] = include_bytes!("../assets/fonts/DejaVuSans.ttf");
const SANS_BOLD: &[u8] = include_bytes!("../assets/fonts/DejaVuSans-Bold.ttf");

/// 注册 "sans-serif" 家族的常规与粗体。幂等，绘图前调用一次即可。
pub fn init() {
    let _ = register_font("sans-serif", plotters::style::FontStyle::Normal, SANS_REGULAR);
    let _ = register_font("sans-serif", plotters::style::FontStyle::Bold, SANS_BOLD);
}
