#!/bin/bash
# Helper: activates venv_gpu and fixes LD_LIBRARY_PATH so qiskit-aer-gpu finds
# the venv's nvjitlink (system CUDA 12.8 lacks the 12.9 symbols cusparse needs).
# Usage:  . experiments/activate_gpu.sh    (must be sourced, not executed)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
. "${ROOT}/venv_gpu/bin/activate"
VENV_NVIDIA="${ROOT}/venv_gpu/lib/python3.12/site-packages/nvidia"
# Prepend ALL venv nvidia lib dirs so system /usr/local/cuda-12.8 libs don't shadow them.
# System CUDA 12.8 is missing symbols that venv's 12.9 nvidia-* libs expect.
CUDA_DIRS=""
for sub in nvjitlink cublas cusparse cusolver cudnn cuda_runtime curand cufft cufile cusolverMg cusparselt nccl nvml nvshmem; do
    if [ -d "${VENV_NVIDIA}/${sub}/lib" ]; then
        CUDA_DIRS="${VENV_NVIDIA}/${sub}/lib:${CUDA_DIRS}"
    fi
done
export LD_LIBRARY_PATH="${CUDA_DIRS}${LD_LIBRARY_PATH}"
echo "GPU env active: venv_gpu + venv CUDA libs prepended to LD_LIBRARY_PATH"
