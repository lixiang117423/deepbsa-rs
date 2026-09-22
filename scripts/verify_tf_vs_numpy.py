#!/usr/bin/env python3
"""TF 官方 model.predict vs numpy 参考实现对比 —— DL 正确性锚点。

从 demo test.csv 取第一条染色体，构造与原版 get_data 相同的窗口输入，
对比两个实现的输出。
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/lixiang/Downloads/DeepBSA-rs/scripts")
from dl_reference import forward, load

MODEL = "/Users/lixiang/Downloads/DeepBSA_linux_v1.4/bin/Models/row_finetune2pool.h5"

data = pd.read_csv("/tmp/deepbsa_orig/demo/test.csv", header=None)
chrome = data[data[0] == data[0].unique()[0]]
freq = np.array([np.array(r[4::2]) / (np.array(r[4::2]) + np.array(r[5::2])) for _, r in chrome.iterrows()])
print("chr1 matrix:", freq.shape)

# 与原版 get_data("DL") 相同的填充与分窗
num_pools = freq.shape[1]
a = len(freq) % 64
pad = np.array([[0] * num_pools] * (64 - a))
padded = np.concatenate((freq, pad), axis=0)
batch = len(padded) // 64
x = np.array([padded[i * 64:(i + 1) * 64] for i in range(batch)]).astype("float64")
print("windows:", x.shape)

cfg, w = load(MODEL)

import tensorflow as tf
model = tf.keras.models.load_model(MODEL)
y_tf = model.predict(x, verbose=0)
print("tf output:", y_tf.shape, "range", y_tf.min(), y_tf.max())

y_np = forward(cfg, w, x.astype(np.float32))
print("numpy output:", y_np.shape)

err = np.abs(y_tf - y_np)
print("max abs err:", err.max(), "mean abs err:", err.mean())
assert err.max() < 1e-5, "numpy reference deviates from TF"
print("PARITY OK: numpy reference == TF")

np.save("/tmp/dl_test_input.npy", x)
np.save("/tmp/dl_test_output.npy", y_tf)
