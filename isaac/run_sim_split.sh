#!/bin/bash
# Split version of run_sim.sh for RTX 50-series GPUs (see README, "Setup on RTX 50-series (Blackwell) GPUs").
# Stage 1 ($UGG_ENV, Python 3.10): E_pen and q1 for all objects in one process.
# Stage 2 ($ISAAC_ENV, Python 3.8, Isaac Gym): simulation, restarted every 25 objects
# because Isaac Gym breaks after ~30 resets (docs/faq.md).
# Run from the repository root:
#   bash isaac/run_sim_split.sh <TEST_FILE>
# where TEST_FILE is the .pk written by model/ugg/eval_generation.py. Results go to
#   results/ugg_generation/ugg_bs8x256_lr0002_l2_cosine/test_obj2hand/sim_on_<TEST_FILE name without the time stamp>/

test_file=$1
[ -f "$test_file" ] || { echo "usage: bash isaac/run_sim_split.sh <TEST_FILE>"; exit 1; }
UGG_ENV=${UGG_ENV:-ugg}
ISAAC_ENV=${ISAAC_ENV:-ugg-isaac}

name=$(basename "$test_file" .pk)
f="sim_on_${name%_????-??-??-??-??-??}"
out_dir="results/ugg_generation/ugg_bs8x256_lr0002_l2_cosine/test_obj2hand/$f"
stp=25

# TEST_FILE is always given explicitly (simulation_test.py picks the newest .pk when it is empty)
mkdir -p "$out_dir"
cfg_test_file="$out_dir/test_file.yaml"
echo "TEST_FILE: $test_file" > "$cfg_test_file"

source "$(dirname "$CONDA_EXE")/../etc/profile.d/conda.sh"

conda activate $UGG_ENV || exit 1
total=$(python -c "import pickle; print(len(pickle.load(open('$test_file', 'rb'))))")
echo "TEST_FILE: $test_file ($total objects), output: $out_dir"
python isaac/sim_split_metrics.py --cfg experiments/ugg_simulation_test.yaml --cfg "$cfg_test_file" \
    --folder=$f --start=0 --step=$total || exit 1
metrics_file=$(ls -t $out_dir/metrics_st_0_*.pk | head -1)
echo "metrics file: $metrics_file"
conda deactivate

conda activate $ISAAC_ENV || exit 1
# Isaac Gym needs ninja and the env's libpython3.8
export PATH=$CONDA_PREFIX/bin:$PATH LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

failed=""
i=0
while [ $i -lt $total ]; do
    full_command="python isaac/sim_split_isaac.py --metrics_file=$metrics_file --out_dir=$out_dir --start=$i --step=$stp"

    echo "Running command: $full_command"

    eval $full_command || failed="$failed $i"

    i=$((i + stp))
done

if [ -n "$failed" ]; then
    echo "FAILED chunks (start index):$failed"
    exit 1
fi
echo "All chunks done."
