#!/bin/bash

# ============================================================
# CD-MAPPO vdo_alpha probe: vdo_alpha=0.1
# Usage: bash cd_mappo_vdo_alpha_01.sh <seed>
# Training on campus UE=50, testing across campus scales
# Results are saved separately from the default CD-MAPPO run.
# ============================================================

SEED=${1:?"Usage: bash cd_mappo_vdo_alpha_01.sh <seed>"}
EXP="experiment_seed_${SEED}"
VDO_ALPHA=0.1
METHOD="cd_mappo_vdo_alpha_01"
MODEL_NAME="${METHOD}_campus50"
SAVE_DIR="./${EXP}/${METHOD}"
MODEL_PATH="${SAVE_DIR}/${MODEL_NAME}/${MODEL_NAME}_final.pth"
COMMON_ARGS="--vdo_alpha ${VDO_ALPHA} --lambda_cf 0.5 --K_actor 4 --deba_memory_size 5000 --deba_sample_size 64 --n_heads 4"

# ---------- Training ----------

python train.py --agent cd_mappo --model_name "${MODEL_NAME}" \
   --scenario_path campus_scenarios/campus_max_ue_50 \
   --save_path "${SAVE_DIR}" --seed "${SEED}" \
   ${COMMON_ARGS}

# ---------- Testing Across Scales ----------

run_test() {
    local scenario_path=$1
    local max_n=${2:-}

    if [ -n "${max_n}" ]; then
        python test_with_action.py --agent cd_mappo --model_name "${MODEL_NAME}" \
            --model_path "${MODEL_PATH}" \
            --scenario_path "${scenario_path}" \
            --save_path "${SAVE_DIR}" \
            ${COMMON_ARGS} --max_n "${max_n}"
    else
        python test_with_action.py --agent cd_mappo --model_name "${MODEL_NAME}" \
            --model_path "${MODEL_PATH}" \
            --scenario_path "${scenario_path}" \
            --save_path "${SAVE_DIR}" \
            ${COMMON_ARGS}
    fi
}

run_test campus_scenarios/campus_max_ue_30
run_test campus_scenarios/campus_max_ue_50
run_test campus_scenarios/campus_max_ue_80
run_test campus_scenarios/campus_max_ue_100
run_test campus_scenarios/campus_max_ue_120 120
run_test campus_scenarios/campus_max_ue_150 150
run_test campus_scenarios/campus_max_ue_200 200

echo "All CD-MAPPO vdo_alpha=${VDO_ALPHA} experiments done for seed=${SEED}."
