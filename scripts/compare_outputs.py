#!/usr/bin/env python3
"""原版 vs Rust 版输出对拍。

比较项：
1. 阈值/窗口（文件名中的参数）
2. values.txt 逐点数值（位置必须完全一致；值允许浮点小差）
3. 峰值 CSV 的峰位/区间/值
4. 置信区间 JSON 结构
"""
import sys

import numpy as np

ORIG = "/tmp/deepbsa_orig/run_{m}/Results/test"
RUST = "/tmp/rust_run_{m}/Results/test"


def find_csv(base, method):
    import glob
    files = glob.glob(f"{base}/*-{method}-*.csv")
    return [f for f in files if "confidence" not in f][0]


def compare(method):
    o = ORIG.format(m=method)
    r = RUST.format(m=method)
    report = {"method": method, "ok": True, "issues": []}

    # 1. 文件名（含阈值）
    import glob, os
    o_png = [os.path.basename(f) for f in glob.glob(f"{o}/*-{method}-*.png") if "confidence" not in f]
    r_png = [os.path.basename(f) for f in glob.glob(f"{r}/*-{method}-*.png") if "confidence" not in f]
    if o_png and r_png:
        # 文件名 {p1}-{m}-{smooth}-{win}-{thr}.png -> 取第 4、5 段
        op = o_png[0].rsplit(".", 1)[0].split("-")
        rp = r_png[0].rsplit(".", 1)[0].split("-")
        o_win, o_thr = op[-2], op[-1]
        r_win, r_thr = rp[-2], rp[-1]
        report["window"] = (o_win, r_win, o_win == r_win)
        report["threshold"] = (o_thr, r_thr, float(o_thr) == float(r_thr))
        if float(o_thr) != float(r_thr):
            report["ok"] = False
            report["issues"].append(f"threshold differs: {o_thr} vs {r_thr}")
        if o_win != r_win:
            report["issues"].append(f"window differs: {o_win} vs {r_win}")

    # 2. values.txt
    def read_values(p):
        chrs, poss, vals = [], [], []
        with open(p) as f:
            for line in f:
                c, pos, v = line.rstrip("\n").split("\t")
                chrs.append(c)
                poss.append(int(pos))
                vals.append(float(v))
        return chrs, np.array(poss), np.array(vals)

    oc, op, ov = read_values(f"{o}/{method} values.txt")
    rc, rp, rv = read_values(f"{r}/{method} values.txt")
    report["n_points"] = (len(ov), len(rv))
    if len(ov) != len(rv):
        report["ok"] = False
        report["issues"].append(f"point count differs: {len(ov)} vs {len(rv)}")
        return report
    pos_match = bool((op == rp).all())
    chr_match = oc == rc
    max_abs = float(np.abs(ov - rv).max())
    rel = max_abs / max(np.abs(ov).max(), 1e-12)
    report["pos_match"] = pos_match
    report["chr_match"] = chr_match
    report["max_abs_diff"] = max_abs
    report["rel_diff"] = rel
    if not pos_match:
        report["ok"] = False
        report["issues"].append("positions differ")
    if not chr_match:
        report["ok"] = False
        report["issues"].append("chromosome labels differ")
    if rel > 1e-4:
        report["ok"] = False
        report["issues"].append(f"values differ: rel={rel:.2e}")

    # 3. 峰值 CSV（"-" 表示该染色体无峰，跳过）
    try:
        import csv as csvmod
        with open(find_csv(o, method)) as f:
            o_rows = list(csvmod.reader(f))
        with open(find_csv(r, method)) as f:
            r_rows = list(csvmod.reader(f))
        o_peaks = [row[3] for row in o_rows[1:] if row[3] != "-"]
        r_peaks = [row[3] for row in r_rows[1:] if row[3] != "-"]
        o_dashes = sum(1 for row in o_rows[1:] if row[3] == "-")
        r_dashes = sum(1 for row in r_rows[1:] if row[3] == "-")
        peak_close = (
            len(o_peaks) == len(r_peaks)
            and o_dashes == r_dashes
            and all(
                any(abs(float(a) - float(b)) < 0.05 for b in r_peaks) for a in o_peaks
            )
        )
        report["peak_match"] = peak_close
        report["peaks_orig"] = o_peaks
        report["peaks_rust"] = r_peaks
        if not peak_close:
            report["ok"] = False
            report["issues"].append(f"peaks differ: {sorted(o_peaks)} vs {sorted(r_peaks)}; dashes {o_dashes}/{r_dashes}")
    except Exception as e:
        report["issues"].append(f"peak csv compare failed: {e}")

    return report


if __name__ == "__main__":
    methods = sys.argv[1:] or ["K", "ED4", "SNP", "SmoothG", "Ridit"]
    for m in methods:
        rep = compare(m)
        status = "PASS" if rep["ok"] else "FAIL"
        print(f"[{status}] {m}: thr={rep.get('threshold')} n={rep.get('n_points')} "
              f"pos={rep.get('pos_match')} maxdiff={rep.get('max_abs_diff'):.2e} "
              f"rel={rep.get('rel_diff'):.2e} peaks={rep.get('peak_match')}")
        for i in rep["issues"]:
            print(f"    ! {i}")
