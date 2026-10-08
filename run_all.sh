#!/bin/zsh
# Full pipeline for one model. usage: ./run_all.sh Qwen/Qwen2.5-1.5B-Instruct [skip_base]
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}
M=$1; S=${M##*/}; mkdir -p runs
[[ "$2" == skip_base ]] || $PY run_model.py $M base > runs/log_${S}_base.txt 2>&1
$PY make_dirs.py $S
for c in prompt_debias proxy; do $PY run_model.py $M $c > runs/log_${S}_$c.txt 2>&1; done
for t in steer_lstar steer_all; do $PY run_model.py $M steer $t > runs/log_${S}_$t.txt 2>&1; done
echo ALL_DONE $S
