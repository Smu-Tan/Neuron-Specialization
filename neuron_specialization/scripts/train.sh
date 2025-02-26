#!/usr/bin/env bash

# repo_dir: root directory of the project
repo_dir="$( cd "$( dirname "$0" )" && pwd )"/..
cd "${repo_dir}"
echo "==== Working directory: ====" >&2
echo "${repo_dir}" >&2
echo "============================" >&2


export MKL_THREADING_LAYER=GNU
export PYTHONPATH="."
python3 toolbox/train.py $@
