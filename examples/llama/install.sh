#!/bin/bash
# Shared install script — called from clusters/<name>/install.pbs.
# Prerequisites (done by the PBS wrapper):
#   1. module load / conda activate
#   2. pip install torch
#   3. export CUDA_HOME, CPATH, LIBRARY_PATH, LD_LIBRARY_PATH
set -euo pipefail

cd $HOME/Megatron-LM

echo "Install nvidia-resiliency-ext"
if [ ! -d "nvidia-resiliency-ext" ]; then
    echo "Cloning"
    git clone https://github.com/NVIDIA/nvidia-resiliency-ext.git -b v0.5.0
fi
cd nvidia-resiliency-ext
pip install poetry pybind11
pip install -v --no-build-isolation -e . 2>&1 | tee -a ../install.log
cd ..

echo "Install megatron-core"
pip install -v --no-build-isolation -e . 2>&1 | tee -a install.log

echo "Install dependencies"
pip install uv cmake
uv pip install -v --no-build-isolation --group build 2>&1 | tee -a install.log
uv pip install -v --no-build-isolation ".[training,dev]" 2>&1 | tee -a install.log

echo "Install apex"
if [ ! -d "apex" ]; then
    echo "Cloning apex"
    git clone https://github.com/NVIDIA/apex.git -b 25.09
    # nvcc and torch version mismatch
    sed -i '84,92s/^/#/' apex/setup.py
    ${APEX_SED_EXTRA:-true}
fi

cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" ./ 2>&1 | tee -a ../install.log
cd ..
