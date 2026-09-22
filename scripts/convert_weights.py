#!/usr/bin/env python3
"""把 Keras .h5 模型转换为 Rust 自定义权重格式 (.bin)。

格式（小端）：
  magic "DBW1" | u32 num_layers
  每层:
    u32 op        0=conv1d 1=maxpool1d 2=upsample1d 3=add 4=input 5=dropout
    u32 n_inputs, u32 * n_inputs   # 入边层索引
    op 参数:
      conv1d: u32 kernel, u32 pad(0=same,1=valid), u32 act(0=none,1=relu,2=sigmoid)
              u32 n_arrays; 每个数组: u32 ndim, u32*ndim dims, f32 data
      maxpool: u32 pool, u32 stride
      upsample: u32 size
"""
import json
import struct
import sys
from pathlib import Path

import h5py
import numpy as np

OP_CONV, OP_POOL, OP_UP, OP_ADD, OP_INPUT, OP_DROPOUT = range(6)
ACT = {"linear": 0, "relu": 1, "sigmoid": 2, None: 0}


def collect_weights(f):
    """收集 model_weights 下所有 (名称 -> 数组)"""
    out = {}

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            out[name] = np.array(obj, dtype=np.float32)

    g = f["model_weights"]
    g.visititems(visit)
    # visititems 给出相对路径，形如 conv1d/conv1d/kernel:0 -> 规范为按层名索引
    by_layer = {}
    for k, v in out.items():
        parts = k.split("/")
        by_layer.setdefault(parts[0], {})[parts[-1]] = v
    return by_layer


def convert(h5_path, out_path):
    with h5py.File(h5_path, "r") as f:
        cfg = json.loads(f.attrs["model_config"])
        weights = collect_weights(f)

    layers = cfg["config"]["layers"]
    name2idx = {l["config"]["name"]: i for i, l in enumerate(layers)}

    def resolve(name):
        # Keras 2.x 中 TensorFlowOpLayer 的入边用图节点名 'tf_op_layer_X'，
        # 而层配置名是 'X' —— 先精确匹配，再做别名剥离
        if name in name2idx:
            return name2idx[name]
        alias = name[len("tf_op_layer_"):] if name.startswith("tf_op_layer_") else name
        if alias in name2idx:
            return name2idx[alias]
        raise KeyError(f"unresolved inbound layer name: {name}")

    buf = bytearray(b"DBW1")
    buf += struct.pack("<I", len(layers))

    for layer in layers:
        cls = layer["class_name"]
        c = layer["config"]
        # 入边
        inbounds = layer.get("inbound_nodes") or []
        srcs = []
        if inbounds:
            for node in inbounds[0]:
                srcs.append(resolve(node[0]))
        if cls == "InputLayer":
            op = OP_INPUT
        elif cls == "Conv1D":
            op = OP_CONV
        elif cls == "MaxPooling1D":
            op = OP_POOL
        elif cls == "UpSampling1D":
            op = OP_UP
        elif cls == "Dropout":
            op = OP_DROPOUT
        elif cls == "TensorFlowOpLayer":
            node_op = c.get("node_def", {}).get("op")
            if node_op == "AddV2":
                op = OP_ADD
            else:
                raise ValueError(f"unsupported TF op: {node_op}")
        else:
            raise ValueError(f"unsupported layer class: {cls}")
        buf += struct.pack("<I", op)
        buf += struct.pack("<I", len(srcs))
        for s in srcs:
            buf += struct.pack("<I", s)
        if op == OP_CONV:
            kernel_size = c["kernel_size"][0]
            pad = 1 if c["padding"] == "valid" else 0
            act = ACT[c.get("activation")]
            buf += struct.pack("<III", kernel_size, pad, act)
            w = weights[c["name"]]
            arrays = [w["kernel:0"]]
            if c.get("use_bias"):
                arrays.append(w["bias:0"])
            buf += struct.pack("<I", len(arrays))
            for a in arrays:
                buf += struct.pack("<I", a.ndim)
                for d in a.shape:
                    buf += struct.pack("<I", d)
                buf += a.astype("<f4").tobytes()
        elif op == OP_POOL:
            buf += struct.pack("<II", c["pool_size"][0], c["strides"][0])
        elif op == OP_UP:
            buf += struct.pack("<I", c["size"])

    Path(out_path).write_bytes(buf)
    print(f"{h5_path} -> {out_path}  ({len(buf)/1e6:.1f} MB, {len(layers)} layers)")


if __name__ == "__main__":
    models_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    for n in range(2, 11):
        convert(models_dir / f"row_finetune{n}pool.h5", out_dir / f"{n}pool.bin")
