#!/bin/bash
#SBATCH --job-name=ldo_v3_reconfirm
#SBATCH --partition=workq
#SBATCH --account=loni_latent_pde
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
#SBATCH --output=/ddnB/work/jing/logs/%j_v3_reconfirm.out
#SBATCH --error=/ddnB/work/jing/logs/%j_v3_reconfirm.err

module purge
module load openfoam
export PATH=$PATH:/home/jing/.local/bin:/home/jing/OpenFOAM/jing-v2212/platforms/linux64GccDPInt32Opt/bin

WORK_DIR=/work/jing
BASE_CASE=/work/jing/base_case
# Deliberately a NEW directory, not overwriting the original ldo_data_v3 --
# once this reconfirmation run is validated (spot-check a sample or two
# against the original's known pressure range [19538, 35294]), you can
# either keep both and cite this one, or rename this over the original.
# Keeping both for now means nothing is lost if something looks wrong.
ENSEMBLE_DIR=/work/jing/ensemble_ch3_reconfirm
DATA_DIR=/work/jing/ldo_data_v3_reconfirm
LOG_DIR=/ddnB/work/jing/logs

mkdir -p ${DATA_DIR} ${LOG_DIR}

N_SAMPLES=${N_SAMPLES:-1000}

LAST_SAMPLE_ID=$(printf "%04d" $((N_SAMPLES - 1)))
if [ ! -d "${ENSEMBLE_DIR}/sample_${LAST_SAMPLE_ID}" ]; then
    echo "Generating ${N_SAMPLES} variable-well cases (reconfirmation) -> ${ENSEMBLE_DIR}"
    python3 ${WORK_DIR}/generate_samples_v3_reconfirm.py \
        --n ${N_SAMPLES} --base ${BASE_CASE} --out ${ENSEMBLE_DIR} --seed 0
    if [ $? -ne 0 ]; then
        echo "FATAL: generate_samples_v3_reconfirm.py failed -- aborting before any solves."
        exit 1
    fi
else
    echo "Ensemble already generated (found sample_${LAST_SAMPLE_ID}) -- skipping generation."
fi

START=${START:-0}
END=${END:-$((N_SAMPLES - 1))}

run_case() {
    i=$1
    SAMPLE_ID=$(printf "%04d" ${i})
    CASE_DIR=${ENSEMBLE_DIR}/sample_${SAMPLE_ID}
    NPY=${DATA_DIR}/sample_${SAMPLE_ID}_p.npy

    if [ ! -d "${CASE_DIR}" ]; then
        echo "ERROR: ${CASE_DIR} does not exist"
        return
    fi
    if [ -f "${NPY}" ]; then echo "SKIP sample_${SAMPLE_ID}"; return; fi

    echo "START sample_${SAMPLE_ID}"
    cd ${CASE_DIR}

    blockMesh >> solver.log 2>&1 || { echo "ERROR: blockMesh failed for sample_${SAMPLE_ID}"; touch FAILED_MESH; return; }
    topoSet   >> solver.log 2>&1 || { echo "ERROR: topoSet failed for sample_${SAMPLE_ID}"; touch FAILED_MESH; return; }
    darcyFoam >> solver.log 2>&1 || { echo "ERROR: solver failed for sample_${SAMPLE_ID}"; touch FAILED_SOLVER; return; }
    if ! grep -q "^End$" solver.log; then
        echo "ERROR: no End for sample_${SAMPLE_ID}"; touch FAILED_SOLVER; return
    fi

    python3 ${WORK_DIR}/extract_fields_v3.py \
        --case ${CASE_DIR} --sample ${i} --out ${DATA_DIR} \
        > ${CASE_DIR}/extract.log 2>&1 || {
            echo "ERROR: extraction failed for sample_${SAMPLE_ID}"
            touch FAILED_EXTRACT; return
        }

    echo "DONE sample_${SAMPLE_ID}"
}
export -f run_case
export WORK_DIR ENSEMBLE_DIR DATA_DIR PATH

MAX_JOBS=${SLURM_NTASKS:-64}
echo "Parallel workers: ${MAX_JOBS}"
echo "Range: sample_${START} .. sample_${END}"

for ((i=START; i<=END; i++)); do
    run_case $i &
    while (( $(jobs -r | wc -l) >= MAX_JOBS )); do sleep 10; done
done
wait

echo "========================================"
echo "Reconfirmation batch ${START}-${END} finished: $(date)"
n_done=$(ls ${DATA_DIR}/sample_*_p.npy 2>/dev/null | wc -l)
n_fail_mesh=$(ls ${ENSEMBLE_DIR}/sample_*/FAILED_MESH 2>/dev/null | wc -l)
n_fail_solve=$(ls ${ENSEMBLE_DIR}/sample_*/FAILED_SOLVER 2>/dev/null | wc -l)
n_fail_extract=$(ls ${ENSEMBLE_DIR}/sample_*/FAILED_EXTRACT 2>/dev/null | wc -l)
echo "Extracted: ${n_done}   Failed mesh: ${n_fail_mesh}   Failed solve: ${n_fail_solve}   Failed extract: ${n_fail_extract}"
echo "========================================"
