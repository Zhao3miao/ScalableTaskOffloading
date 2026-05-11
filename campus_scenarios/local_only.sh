#!/bin/bash

# ============================================================
# Local-only heuristic baseline
# Usage: bash local_only.sh <seed>
# Testing on Campus scales
# ============================================================

SEED=${1:?"Usage: bash local_only.sh <seed>"}
EXP="experiment_seed_${SEED}"
METHOD="local_only"
MODEL="local_only_campus50"

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

echo "All LOCAL_ONLY experiments done for seed=${SEED}."
