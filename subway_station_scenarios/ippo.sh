#!/bin/bash

# ============================================================
# IPPO: independent PPO baseline
# Usage: bash ippo.sh <seed>
# Training on Subway Station UE=50, testing across subway scales
# ============================================================

SEED=${1:?"Usage: bash ippo.sh <seed>"}
EXP="experiment_seed_${SEED}"

# ---------- Training ----------

python train.py --agent ippo --model_name ippo_subway50 \
   --scenario_path subway_station_scenarios/subway_station_max_ue_50 \
   --save_path ./${EXP}/ippo --seed ${SEED}

# ---------- Testing Across Scales ----------

python test_with_action.py --agent ippo --model_name ippo_subway50 \
   --model_path ./${EXP}/ippo/ippo_subway50/ippo_subway50_final.pth \
   --scenario_path subway_station_scenarios/subway_station_max_ue_30 \
   --save_path ./${EXP}/ippo

python test_with_action.py --agent ippo --model_name ippo_subway50 \
   --model_path ./${EXP}/ippo/ippo_subway50/ippo_subway50_final.pth \
   --scenario_path subway_station_scenarios/subway_station_max_ue_50 \
   --save_path ./${EXP}/ippo

python test_with_action.py --agent ippo --model_name ippo_subway50 \
   --model_path ./${EXP}/ippo/ippo_subway50/ippo_subway50_final.pth \
   --scenario_path subway_station_scenarios/subway_station_max_ue_80 \
   --save_path ./${EXP}/ippo

python test_with_action.py --agent ippo --model_name ippo_subway50 \
   --model_path ./${EXP}/ippo/ippo_subway50/ippo_subway50_final.pth \
   --scenario_path subway_station_scenarios/subway_station_max_ue_100 \
   --save_path ./${EXP}/ippo

python test_with_action.py --agent ippo --model_name ippo_subway50 \
    --model_path ./${EXP}/ippo/ippo_subway50/ippo_subway50_final.pth \
    --scenario_path subway_station_scenarios/subway_station_max_ue_120 \
    --save_path ./${EXP}/ippo --max_n 120

python test_with_action.py --agent ippo --model_name ippo_subway50 \
    --model_path ./${EXP}/ippo/ippo_subway50/ippo_subway50_final.pth \
    --scenario_path subway_station_scenarios/subway_station_max_ue_150 \
    --save_path ./${EXP}/ippo --max_n 150

python test_with_action.py --agent ippo --model_name ippo_subway50 \
    --model_path ./${EXP}/ippo/ippo_subway50/ippo_subway50_final.pth \
    --scenario_path subway_station_scenarios/subway_station_max_ue_200 \
    --save_path ./${EXP}/ippo --max_n 200

echo "All IPPO experiments done for seed=${SEED}."
