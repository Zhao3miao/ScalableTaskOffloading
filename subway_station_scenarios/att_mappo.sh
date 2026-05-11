#!/bin/bash

# ============================================================
# ATT-MAPPO: blind actor + attention critic
# Usage: bash att_mappo.sh <seed>
# Training on Subway Station UE=50, testing across subway scales
# ============================================================

SEED=${1:?"Usage: bash att_mappo.sh <seed>"}
EXP="experiment_seed_${SEED}"

# ---------- Training ----------

python train.py --agent att_mappo --model_name att_mappo_subway50 \
   --scenario_path subway_station_scenarios/subway_station_max_ue_50 \
   --save_path ./${EXP}/att_mappo --seed ${SEED}

# ---------- Testing Across Scales ----------

python test_with_action.py --agent att_mappo --model_name att_mappo_subway50 \
   --model_path ./${EXP}/att_mappo/att_mappo_subway50/att_mappo_subway50_final.pth \
   --scenario_path subway_station_scenarios/subway_station_max_ue_30 \
   --save_path ./${EXP}/att_mappo

python test_with_action.py --agent att_mappo --model_name att_mappo_subway50 \
   --model_path ./${EXP}/att_mappo/att_mappo_subway50/att_mappo_subway50_final.pth \
   --scenario_path subway_station_scenarios/subway_station_max_ue_50 \
   --save_path ./${EXP}/att_mappo

python test_with_action.py --agent att_mappo --model_name att_mappo_subway50 \
   --model_path ./${EXP}/att_mappo/att_mappo_subway50/att_mappo_subway50_final.pth \
   --scenario_path subway_station_scenarios/subway_station_max_ue_80 \
   --save_path ./${EXP}/att_mappo

python test_with_action.py --agent att_mappo --model_name att_mappo_subway50 \
   --model_path ./${EXP}/att_mappo/att_mappo_subway50/att_mappo_subway50_final.pth \
   --scenario_path subway_station_scenarios/subway_station_max_ue_100 \
   --save_path ./${EXP}/att_mappo

python test_with_action.py --agent att_mappo --model_name att_mappo_subway50 \
    --model_path ./${EXP}/att_mappo/att_mappo_subway50/att_mappo_subway50_final.pth \
    --scenario_path subway_station_scenarios/subway_station_max_ue_120 \
    --save_path ./${EXP}/att_mappo --max_n 120

python test_with_action.py --agent att_mappo --model_name att_mappo_subway50 \
    --model_path ./${EXP}/att_mappo/att_mappo_subway50/att_mappo_subway50_final.pth \
    --scenario_path subway_station_scenarios/subway_station_max_ue_150 \
    --save_path ./${EXP}/att_mappo --max_n 150

python test_with_action.py --agent att_mappo --model_name att_mappo_subway50 \
    --model_path ./${EXP}/att_mappo/att_mappo_subway50/att_mappo_subway50_final.pth \
    --scenario_path subway_station_scenarios/subway_station_max_ue_200 \
    --save_path ./${EXP}/att_mappo --max_n 200

echo "All ATT_MAPPO experiments done for seed=${SEED}."
