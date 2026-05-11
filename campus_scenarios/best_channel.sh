#!/bin/bash

# ============================================================
# Best-channel / nearest-edge heuristic baseline
# Usage: bash best_channel.sh <seed>
# Testing on Campus scales
# ============================================================

SEED=${1:?"Usage: bash best_channel.sh <seed>"}
EXP="experiment_seed_${SEED}"
METHOD="best_channel"
MODEL="best_channel_campus50"

python test_rule_based.py --agent ${METHOD} --model_name ${MODEL} \
   --scenario_path campus_scenarios/campus_max_ue_30 \
   --save_path ./${EXP}/${METHOD} --seed ${SEED}

python test_rule_based.py --agent ${METHOD} --model_name ${MODEL} \
   --scenario_path campus_scenarios/campus_max_ue_50 \
   --save_path ./${EXP}/${METHOD} --seed ${SEED}

python test_rule_based.py --agent ${METHOD} --model_name ${MODEL} \
   --scenario_path campus_scenarios/campus_max_ue_80 \
   --save_path ./${EXP}/${METHOD} --seed ${SEED}

python test_rule_based.py --agent ${METHOD} --model_name ${MODEL} \
   --scenario_path campus_scenarios/campus_max_ue_100 \
   --save_path ./${EXP}/${METHOD} --seed ${SEED}

python test_rule_based.py --agent ${METHOD} --model_name ${MODEL} \
   --scenario_path campus_scenarios/campus_max_ue_120 \
   --save_path ./${EXP}/${METHOD} --seed ${SEED} --max_n 120

python test_rule_based.py --agent ${METHOD} --model_name ${MODEL} \
   --scenario_path campus_scenarios/campus_max_ue_150 \
   --save_path ./${EXP}/${METHOD} --seed ${SEED} --max_n 150

python test_rule_based.py --agent ${METHOD} --model_name ${MODEL} \
   --scenario_path campus_scenarios/campus_max_ue_200 \
   --save_path ./${EXP}/${METHOD} --seed ${SEED} --max_n 200

echo "All BEST_CHANNEL experiments done for seed=${SEED}."
