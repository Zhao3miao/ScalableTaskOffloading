#!/bin/bash

# ============================================================
# Random edge-only diagnostic baseline
# Usage: bash random_edge_only.sh <seed>
# Testing on Subway Station scales
# ============================================================

SEED=${1:?"Usage: bash random_edge_only.sh <seed>"}
EXP="experiment_seed_${SEED}"
METHOD="random_edge_only"
MODEL="random_edge_only_subway50"

python test_rule_based.py --agent ${METHOD} --model_name ${MODEL} \
   --scenario_path subway_station_scenarios/subway_station_max_ue_30 \
   --save_path ./${EXP}/${METHOD} --seed ${SEED}

python test_rule_based.py --agent ${METHOD} --model_name ${MODEL} \
   --scenario_path subway_station_scenarios/subway_station_max_ue_50 \
   --save_path ./${EXP}/${METHOD} --seed ${SEED}

python test_rule_based.py --agent ${METHOD} --model_name ${MODEL} \
   --scenario_path subway_station_scenarios/subway_station_max_ue_80 \
   --save_path ./${EXP}/${METHOD} --seed ${SEED}

python test_rule_based.py --agent ${METHOD} --model_name ${MODEL} \
   --scenario_path subway_station_scenarios/subway_station_max_ue_100 \
   --save_path ./${EXP}/${METHOD} --seed ${SEED}

python test_rule_based.py --agent ${METHOD} --model_name ${MODEL} \
   --scenario_path subway_station_scenarios/subway_station_max_ue_120 \
   --save_path ./${EXP}/${METHOD} --seed ${SEED} --max_n 120

python test_rule_based.py --agent ${METHOD} --model_name ${MODEL} \
   --scenario_path subway_station_scenarios/subway_station_max_ue_150 \
   --save_path ./${EXP}/${METHOD} --seed ${SEED} --max_n 150

python test_rule_based.py --agent ${METHOD} --model_name ${MODEL} \
   --scenario_path subway_station_scenarios/subway_station_max_ue_200 \
   --save_path ./${EXP}/${METHOD} --seed ${SEED} --max_n 200

echo "All RANDOM_EDGE_ONLY subway diagnostics done for seed=${SEED}."
