#!/bin/bash
# 搭建原版 DeepBSA 的参考运行环境（黄金基线对拍用）
# 注意：arm64 mac 没有 tensorflow==2.10.0 轮子，用 tensorflow-macos 替代
set -x
CONDA=/Users/lixiang/miniforge3/condabin/conda
$CONDA create -n deepbsa-ref python=3.9 -y || exit 1
PIP=/Users/lixiang/miniforge3/envs/deepbsa-ref/bin/pip
$PIP install h5py numpy pandas scipy statsmodels matplotlib tqdm
$PIP install tensorflow-macos==2.10.0 || $PIP install tensorflow-macos || $PIP install tensorflow==2.13.1
$PIP install rpy2==3.5.4 || echo "rpy2 FAILED - will use Rscript fallback for span verification"
echo "=== ENV SETUP DONE ==="
/Users/lixiang/miniforge3/envs/deepbsa-ref/bin/python -c "import h5py, numpy, pandas, scipy, statsmodels; print('core deps OK')"
/Users/lixiang/miniforge3/envs/deepbsa-ref/bin/python -c "import tensorflow as tf; print('TF', tf.__version__)" || echo "TF NOT AVAILABLE"
