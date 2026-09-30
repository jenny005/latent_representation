#!/bin/bash
#SBATCH --job-name=ch3_full_sweep_reconfirm
#SBATCH --partition=workq
#SBATCH --account=loni_latent_pde
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH --output=/ddnB/work/jing/logs/%j_ch3_full_sweep_reconfirm.out
#SBATCH --error=/ddnB/work/jing/logs/%j_ch3_full_sweep_reconfirm.err

module purge
export PATH=$PATH:/home/jing/.local/bin

# History: this replaced mismatched n=8/64 numbers (ldo_data_v2, 300
# samples, 20 epochs) and merged them with already-confirmed n=16/32
# numbers into one consistent ldo_data_v3 sweep. THIS VERSION goes one
# step further: ldo_data_v3 has been regenerated as ldo_data_v3_reconfirm
# on the CURRENT solver binary (see darcyFoam_v3_reconfirm.sbatch), so
# Table 3.1's variable-well column and the fixedwell column now provably
# come from the identical compiled darcyFoam, not two solver versions.
#
# NOTE: an earlier version of this script pointed at ldo_data_v3 (the
# original directory) and was run without this edit -- that run reproduced
# bit-identical numbers to the previous confirmed sweep, which turned out
# to mean the path was never actually updated, not that the reconfirmation
# was verified. Confirm this script actually points at ldo_data_v3_reconfirm
# below before trusting any result from it.
python3 /work/jing/pod_vs_ae_eta.py /work/jing/ldo_data_v3_reconfirm \
    --n-dims 8 16 32 64 --held-out-frac 0.15 --max-samples 1000 \
    --epochs 150 --min-epochs 40 --patience 20 --batch-size 256 \
    --num-threads 64 --seed 0 \
    --out /ddnB/work/jing/logs/ch3_full_sweep_reconfirm_results.json

echo "========================================"
echo "Chapter 3 full n=8..64 sweep (reconfirm data) finished: $(date)"
echo "========================================"
