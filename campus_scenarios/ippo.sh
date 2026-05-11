#!/bin/bash

# ============================================================
# IPPO: independent PPO baseline
# Usage: bash ippo.sh <seed>
# Training on Campus UE=50, testing across campus scales
# ============================================================

SEED=${1:?"Usage: bash ippo.sh <seed>"}
EXP="experiment_seed_${SEED}"

# ---------- Training ----------

python train.py --agent ippo --model_name ippo_campus50 \
   --scenario_path campus_scenarios/campus_max_ue_50 \
   --save_path ./${EXP}/ippo --seed ${SEED}

# ---------- Testing Across Scales ----------

python test_with_action.py --agent ippo --model_name ippo_campus50 \
   --model_path ./${EXP}/ippo/ippo_campus50/ippo_campus50_final.pth \
   --scenario_path campus_scenarios/campus_max_ue_30 \
   --save_path ./${EXP}/ippo

python test_with_action.py --agent ippo --model_name ippo_campus50 \
   --model_path ./${EXP}/ippo/ippo_campus50/ippo_campus50_final.pth \
   --scenario_path campus_scenarios/campus_max_ue_50 \
   --save_path ./${EXP}/ippo

python test_with_action.py --agent ippo --model_name ippo_campus50 \
   --model_path ./${EXP}/ippo/ippo_campus50/ippo_campus50_final.pth \
   --scenario_path campus_scenarios/campus_max_ue_80 \
   --save_path ./${EXP}/ippo

python test_with_action.py --agent ippo --model_name ippo_campus50 \
   --model_path ./${EXP}/ippo/ippo_campus50/ippo_campus50_final.pth \
   --scenario_path campus_scenarios/campus_max_ue_100 \
   --save_path ./${EXP}/ippo

python test_with_action.py --agent ippo --model_name ippo_campus50 \
    --model_path ./${EXP}/ippo/ippo_campus50/ippo_campus50_final.pth \
    --scenario_path campus_scenarios/campus_max_ue_120 \
    --save_path ./${EXP}/ippo --max_n 120

python test_with_action.py --agent ippo --model_name ippo_campus50 \
    --model_path ./${EXP}/ippo/ippo_campus50/ippo_campus50_final.pth \
    --scenario_path campus_scenarios/campus_max_ue_150 \
    --save_path ./${EXP}/ippo --max_n 150

python test_with_action.py --agent ippo --model_name ippo_campus50 \
    --model_path ./${EXP}/ippo/ippo_campus50/ippo_campus50_final.pth \
    --scenario_path campus_scenarios/campus_max_ue_200 \
    --save_path ./${EXP}/ippo --max_n 200

echo "All IPPO experiments done for seed=${SEED}."
