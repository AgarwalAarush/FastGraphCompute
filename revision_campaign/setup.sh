#!/usr/bin/env bash
# Explicitly invoked setup only. Run on a build-capable compute node.
set -euo pipefail
action=${1:?usage: setup.sh core|faiss|cuvs|ggnn|build-fgc|build-counters PREFIX_OR_SOURCE}
target=${2:?supply a NEW environment prefix or source directory}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
torch_version=${FGC_TORCH_VERSION:-2.5.1}
torch_wheels=${FGC_TORCH_INDEX:-https://download.pytorch.org/whl/cu124}
arch=${TORCH_CUDA_ARCH_LIST:-9.0}
if [[ $action == build-fgc || $action == build-counters ]]; then
  : "${FGC_CORE_PYTHON:?set FGC_CORE_PYTHON to the core environment interpreter}"
  cd "$target"
  export DEV_MODE=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="$arch"
  export MAX_JOBS=${MAX_JOBS:-2} NUM_PARALLEL_JOBS=${NUM_PARALLEL_JOBS:-2}
  "$FGC_CORE_PYTHON" setup.py build_ext --inplace -j 1
  exit
fi
if [[ -e $target ]]; then
  echo "Refusing to change an existing environment: $target" >&2
  exit 2
fi
command -v conda >/dev/null
case "$action" in
  core|ggnn)
    conda create -y -p "$target" -c conda-forge python=3.11 pip cmake=3.30 setuptools wheel
    ;;
  faiss)
    conda create -y -p "$target" -c pytorch -c nvidia -c conda-forge \
      python=3.11 pip faiss-gpu=1.12.0 cuda-version=12.4 numpy=1.26.4
    ;;
  cuvs)
    # A fixed CUDA-12 generation; never silently choose the newest RAPIDS stack.
    conda create -y -p "$target" -c rapidsai -c conda-forge -c nvidia \
      python=3.11 pip cuvs=25.02 cuda-version=12.4 numpy=1.26.4
    ;;
  *) echo "Unknown setup action: $action" >&2; exit 2 ;;
esac
"$target/bin/python" -m pip install "torch==$torch_version" --index-url "$torch_wheels"
"$target/bin/python" -m pip install -r "$script_dir/requirements.txt"
if [[ $action == ggnn ]]; then
  : "${GGNN_SOURCE:?clone the pinned GGNN source and set GGNN_SOURCE}"
  expected=2d03b36362bc824e37099092eabb95172ec18b64
  [[ $(git -C "$GGNN_SOURCE" rev-parse HEAD) == "$expected" ]]
  export TORCH_CUDA_ARCH_LIST="$arch" CMAKE_BUILD_PARALLEL_LEVEL=2
  "$target/bin/python" -m pip install "$GGNN_SOURCE"
fi
"$target/bin/python" -m pip freeze > "$target/fastgraph-installed-packages.txt"
echo "Created $action environment at $target. Run CUDA preflight before benchmarking."
