#!/bin/bash

# ============================================================
# Blind IPPO: independent PPO baseline
# Usage: bash blind_ippo.sh <seed>
# Training on Subway Station UE=50, testing across subway scales
# ============================================================

SEED=${1:?"Usage: bash blind_ippo.sh <seed>"}
EXP="experiment_seed_${SEED}"

# ---------- Training ----------

python train.py --agent blind_ippo --model_name blind_ippo_subway50 \
   --scenario_path subway_station_scenarios/subway_station_max_ue_50 \
   --save_path ./${EXP}/blind_ippo --seed ${SEED}

# ---------- Testing Across Scales ----------

python test_with_action.py --agent blind_ippo --model_name blind_ippo_subway50 \
   --model_path ./${EXP}/blind_ippo/blind_ippo_subway50/blind_ippo_subway50_final.pth \
   --scenario_path subway_station_scenarios/subway_station_max_ue_30 \
   --save_path ./${EXP}/blind_ippo

python test_with_action.py --agent blind_ippo --model_name blind_ippo_subway50 \
   --model_path ./${EXP}/blind_ippo/blind_ippo_subway50/blind_ippo_subway50_final.pth \
   --scenario_path subway_station_scenarios/subway_station_max_ue_50 \
   --save_path ./${EXP}/blind_ippo

python test_with_action.py --agent blind_ippo --model_name blind_ippo_subway50 \
   --model_path ./${EXP}/blind_ippo/blind_ippo_subway50/blind_ippo_subway50_final.pth \
   --scenario_path subway_station_scenarios/subway_station_max_ue_80 \
   --save_path ./${EXP}/blind_ippo

python test_with_action.py --agent blind_ippo --model_name blind_ippo_subway50 \
   --model_path ./${EXP}/blind_ippo/blind_ippo_subway50/blind_ippo_subway50_final.pth \
   --scenario_path subway_station_scenarios/subway_station_max_ue_100 \
   --save_path ./${EXP}/blind_ippo

python test_with_action.py --agent blind_ippo --model_name blind_ippo_subway50 \
    --model_path ./${EXP}/blind_ippo/blind_ippo_subway50/blind_ippo_subway50_final.pth \
    --scenario_path subway_station_scenarios/subway_station_max_ue_120 \
    --save_path ./${EXP}/blind_ippo --max_n 120

python test_with_action.py --agent blind_ippo --model_name blind_ippo_subway50 \
    --model_path ./${EXP}/blind_ippo/blind_ippo_subway50/blind_ippo_subway50_final.pth \
    --scenario_path subway_station_scenarios/subway_station_max_ue_150 \
    --save_path ./${EXP}/blind_ippo --max_n 150

python test_with_action.py --agent blind_ippo --model_name blind_ippo_subway50 \
    --model_path ./${EXP}/blind_ippo/blind_ippo_subway50/blind_ippo_subway50_final.pth \
    --scenario_path subway_station_scenarios/subway_station_max_ue_200 \
    --save_path ./${EXP}/blind_ippo --max_n 200

echo "All BLIND_IPPO experiments done for seed=${SEED}."
