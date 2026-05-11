#!/bin/bash

# ============================================================
# ATT-MAPPO: blind actor + attention critic (self-attention + congestion encoding)
# Usage: bash att_mappo.sh <seed>
# Training on Campus UE=50, testing across campus scales
# ============================================================

SEED=${1:?"Usage: bash att_mappo.sh <seed>"}
EXP="experiment_seed_${SEED}"

# ---------- Training ----------

python train.py --agent att_mappo --model_name att_mappo_campus50 \
   --scenario_path campus_scenarios/campus_max_ue_50 \
   --save_path ./${EXP}/att_mappo --seed ${SEED}

# ---------- Testing Across Scales ----------

python test_with_action.py --agent att_mappo --model_name att_mappo_campus50 \
   --model_path ./${EXP}/att_mappo/att_mappo_campus50/att_mappo_campus50_final.pth \
   --scenario_path campus_scenarios/campus_max_ue_30 \
   --save_path ./${EXP}/att_mappo

python test_with_action.py --agent att_mappo --model_name att_mappo_campus50 \
   --model_path ./${EXP}/att_mappo/att_mappo_campus50/att_mappo_campus50_final.pth \
   --scenario_path campus_scenarios/campus_max_ue_50 \
   --save_path ./${EXP}/att_mappo

python test_with_action.py --agent att_mappo --model_name att_mappo_campus50 \
   --model_path ./${EXP}/att_mappo/att_mappo_campus50/att_mappo_campus50_final.pth \
   --scenario_path campus_scenarios/campus_max_ue_80 \
   --save_path ./${EXP}/att_mappo

python test_with_action.py --agent att_mappo --model_name att_mappo_campus50 \
   --model_path ./${EXP}/att_mappo/att_mappo_campus50/att_mappo_campus50_final.pth \
   --scenario_path campus_scenarios/campus_max_ue_100 \
   --save_path ./${EXP}/att_mappo

python test_with_action.py --agent att_mappo --model_name att_mappo_campus50 \
    --model_path ./${EXP}/att_mappo/att_mappo_campus50/att_mappo_campus50_final.pth \
    --scenario_path campus_scenarios/campus_max_ue_120 \
    --save_path ./${EXP}/att_mappo --max_n 120

python test_with_action.py --agent att_mappo --model_name att_mappo_campus50 \
    --model_path ./${EXP}/att_mappo/att_mappo_campus50/att_mappo_campus50_final.pth \
    --scenario_path campus_scenarios/campus_max_ue_150 \
    --save_path ./${EXP}/att_mappo --max_n 150

    python test_with_action.py --agent att_mappo --model_name att_mappo_campus50 \
    --model_path ./${EXP}/att_mappo/att_mappo_campus50/att_mappo_campus50_final.pth \
    --scenario_path campus_scenarios/campus_max_ue_200 \
    --save_path ./${EXP}/att_mappo --max_n 200

echo "All ATT_MAPPO experiments done for seed=${SEED}."
