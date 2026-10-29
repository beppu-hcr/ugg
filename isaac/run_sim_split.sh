#!/bin/bash
# Split version of run_sim.sh for RTX 50-series GPUs (see SETUP_NOTES.md).
# Stage 1 (ugg env): E_pen and q1 for all objects in one process.
# Stage 2 (grasp2grasp env, Isaac Gym): simulation, restarted every $stp objects
# because Isaac Gym breaks after ~30 resets (docs/faq.md).
# Run from the repository root.

cfg="experiments/ugg_simulation_test_full.yaml"
f="sim_on_test_obj2hand_full1126"
out_dir="results/ugg_generation/ugg_bs8x256_lr0002_l2_cosine/test_obj2hand/$f"
total=1126
stp=25

source ~/miniconda3/etc/profile.d/conda.sh

conda activate ugg
python isaac/sim_split_metrics.py --cfg $cfg --folder=$f --start=0 --step=$total || exit 1
metrics_file=$(ls -t $out_dir/metrics_st_0_*.pk | head -1)
echo "metrics file: $metrics_file"
conda deactivate

conda activate grasp2grasp
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
