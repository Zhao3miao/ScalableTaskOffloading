#!/bin/bash
set -e

# CD-MAPPO without the deconfounded critic.
# Default seeds: 42, 233, 666.
# Usage:
#   bash ablation_study/experiment_wo_deconf_critic.sh
#   bash ablation_study/experiment_wo_deconf_critic.sh 42
#   bash ablation_study/experiment_wo_deconf_critic.sh 42 233 666

if [ "$#" -gt 0 ]; then
  SEEDS=("$@")
else
  SEEDS=(42 233 666)
fi

SCALES=(30 50 80 100 120 150 200)
METHOD="cd_mappo_wo_deconf_critic"
TRAIN_SCRIPT="ablation_study/train_cd_mappo_wo_deconf_critic.py"
CAMPUS_ARGS="--vdo_alpha 0.7 --lambda_cf 0.5 --K_actor 4 --deba_memory_size 5000 --deba_sample_size 64 --n_heads 4"
SUBWAY_ARGS="--vdo_alpha 0.5 --lambda_cf 0.5 --K_actor 4 --deba_memory_size 5000 --deba_sample_size 64 --n_heads 4"

run_scenario() {
  local seed="$1"
  local model_name="$2"
  local scenario_prefix="$3"
  local cd_args="$4"

  local exp="experiment_seed_${seed}"
  local save_path="./${exp}/${METHOD}"
  local model_path="${save_path}/${model_name}/${model_name}_final.pth"

  echo "============================================================"
  echo "Training ${METHOD} (${model_name}) with seed=${seed}"
  echo "============================================================"

  python "${TRAIN_SCRIPT}" \
    --model_name "${model_name}" \
    --scenario_path "${scenario_prefix}_50" \
    --save_path "${save_path}" \
    --seed "${seed}" \
    ${cd_args}

  echo "============================================================"
  echo "Testing ${METHOD} (${model_name}) with seed=${seed}"
  echo "============================================================"

  for scale in "${SCALES[@]}"; do
    local max_n_arg=""
    if [ "${scale}" -gt 100 ]; then
      max_n_arg="--max_n ${scale}"
    fi

    python test_with_action.py \
      --agent cd_mappo \
      --model_name "${model_name}" \
      --model_path "${model_path}" \
      --scenario_path "${scenario_prefix}_${scale}" \
      --save_path "${save_path}" \
      ${cd_args} \
      ${max_n_arg}
  done
}

for seed in "${SEEDS[@]}"; do
  run_scenario "${seed}" "cd_mappo_wo_deconf_critic_campus50" "campus_scenarios/campus_max_ue" "${CAMPUS_ARGS}"
  run_scenario "${seed}" "cd_mappo_wo_deconf_critic_subway50" "subway_station_scenarios/subway_station_max_ue" "${SUBWAY_ARGS}"
  echo "All ${METHOD} experiments done for seed=${seed}."
done
