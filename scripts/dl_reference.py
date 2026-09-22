#!/usr/bin/env python3
"""numpy 复刻的 DL 前向传播参考实现（对拍黄金基线）。

直接从 .h5 读权重，按模型 config 的图结构执行前向传播，
输出与 TF model.predict 一致的结果，用于验证 Rust 实现。
"""
import json
import sys

import h5py
import numpy as np


def load(h5_path):
    with h5py.File(h5_path, "r") as f:
        cfg = json.loads(f.attrs["model_config"])
        w = {}
        f["model_weights"].visititems(
            lambda name, obj: w.__setitem__(name, np.array(obj)) if isinstance(obj, h5py.Dataset) else None
        )
    return cfg, w


def forward(cfg, w, x):
    """x: (batch, L, C) -> (batch, L, 1)"""
    layers = cfg["config"]["layers"]
    name2idx = {l["config"]["name"]: i for i, l in enumerate(layers)}
    vals = [None] * len(layers)

    def resolve(name):
        # 同 convert_weights.py：'tf_op_layer_X' -> 'X' 别名
        if name in name2idx:
            return name2idx[name]
        alias = name[len("tf_op_layer_"):] if name.startswith("tf_op_layer_") else name
        if alias in name2idx:
            return name2idx[alias]
        raise KeyError(name)

    def get_weights(name):
        out = {}
        for k, v in w.items():
            if k.startswith(name + "/") or k.startswith(name + "/" + name + "/"):
                out[k.split("/")[-1]] = v
        return out

    for i, layer in enumerate(layers):
        cls = layer["class_name"]
        c = layer["config"]
        inbounds = layer.get("inbound_nodes") or []
        srcs = [resolve(node[0]) for node in inbounds[0]] if inbounds else []
        if cls == "InputLayer":
            vals[i] = x
        elif cls == "Dropout":
            vals[i] = vals[srcs[0]]
        elif cls == "TensorFlowOpLayer":
            assert c.get("node_def", {}).get("op") == "AddV2"
            vals[i] = vals[srcs[0]] + vals[srcs[1]]
        elif cls == "MaxPooling1D":
            pool, stride = c["pool_size"][0], c["strides"][0]
            v = vals[srcs[0]]  # (b, L, C)
            b, L, ch = v.shape
            assert c["padding"] == "valid"
            v = v[:, : (L // pool) * pool, :].reshape(b, L // pool, pool, ch)
            vals[i] = v.max(axis=2)
        elif cls == "UpSampling1D":
            v = vals[srcs[0]]
            vals[i] = np.repeat(v, c["size"], axis=1)
        elif cls == "Conv1D":
            v = vals[srcs[0]]
            ksz = c["kernel_size"][0]
            weight = get_weights(c["name"])
            kernel = weight["kernel:0"]  # (k, in, out)
            bias = weight.get("bias:0")
            L = v.shape[1]
            if c["padding"] == "same":
                total = ksz - 1
                left, right = total // 2, total - total // 2
                v = np.pad(v, ((0, 0), (left, right), (0, 0)))
            out_len = v.shape[1] - ksz + 1
            # im2col: (b, out_len, k*in) @ (k*in, out)
            b, padded, ch = v.shape
            cols = np.empty((b, out_len, ksz * ch), dtype=v.dtype)
            for t in range(out_len):
                cols[:, t, :] = v[:, t : t + ksz, :].reshape(b, -1)
            out = cols @ kernel.reshape(ksz * ch, -1)
            if bias is not None:
                out = out + bias
            act = c.get("activation")
            if act == "relu":
                out = np.maximum(out, 0.0)
            elif act == "sigmoid":
                out = 1.0 / (1.0 + np.exp(-out.astype(np.float64)))
            vals[i] = out
        else:
            raise ValueError(cls)
    return vals[-1]


if __name__ == "__main__":
    h5_path, x_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    cfg, w = load(h5_path)
    x = np.load(x_path).astype(np.float32)
    y = forward(cfg, w, x)
    np.save(out_path, y)
    print(f"{h5_path}: input{x.shape} -> output{y.shape} saved to {out_path}")
