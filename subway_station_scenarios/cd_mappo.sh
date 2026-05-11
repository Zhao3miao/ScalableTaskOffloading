#!/bin/bash

# ============================================================
# CD-MAPPO: SharedBlindActor + DeconfoundedCritic
# Usage: bash cd_mappo.sh <seed>
# Training on Subway Station UE=50, testing across subway scales
# ============================================================

SEED=${1:?"Usage: bash cd_mappo.sh <seed>"}
EXP="experiment_seed_${SEED}"

# ---------- Training ----------

python train.py --agent cd_mappo --model_name cd_mappo_subway50 \
   --scenario_path subway_station_scenarios/subway_station_max_ue_50 \
   --save_path ./${EXP}/cd_mappo --seed ${SEED} \
   --vdo_alpha 0.5 --lambda_cf 0.5 --K_actor 4 \
   --deba_memory_size 5000 --deba_sample_size 64 --n_heads 4

# ---------- Testing Across Scales ----------

python test_with_action.py --agent cd_mappo --model_name cd_mappo_subway50 \
   --model_path ./${EXP}/cd_mappo/cd_mappo_subway50/cd_mappo_subway50_final.pth \
   --scenario_path subway_station_scenarios/subway_station_max_ue_30 \
   --save_path ./${EXP}/cd_mappo \
   --vdo_alpha 0.5 --lambda_cf 0.5 --K_actor 4 --deba_memory_size 5000 --deba_sample_size 64 --n_heads 4

python test_with_action.py --agent cd_mappo --model_name cd_mappo_subway50 \
   --model_path ./${EXP}/cd_mappo/cd_mappo_subway50/cd_mappo_subway50_final.pth \
   --scenario_path subway_station_scenarios/subway_station_max_ue_50 \
   --save_path ./${EXP}/cd_mappo \
   --vdo_alpha 0.5 --lambda_cf 0.5 --K_actor 4 --deba_memory_size 5000 --deba_sample_size 64 --n_heads 4

python test_with_action.py --agent cd_mappo --model_name cd_mappo_subway50 \
   --model_path ./${EXP}/cd_mappo/cd_mappo_subway50/cd_mappo_subway50_final.pth \
   --scenario_path subway_station_scenarios/subway_station_max_ue_80 \
   --save_path ./${EXP}/cd_mappo \
   --vdo_alpha 0.5 --lambda_cf 0.5 --K_actor 4 --deba_memory_size 5000 --deba_sample_size 64 --n_heads 4

python test_with_action.py --agent cd_mappo --model_name cd_mappo_subway50 \
   --model_path ./${EXP}/cd_mappo/cd_mappo_subway50/cd_mappo_subway50_final.pth \
   --scenario_path subway_station_scenarios/subway_station_max_ue_100 \
   --save_path ./${EXP}/cd_mappo \
   --vdo_alpha 0.5 --lambda_cf 0.5 --K_actor 4 --deba_memory_size 5000 --deba_sample_size 64 --n_heads 4

python test_with_action.py --agent cd_mappo --model_name cd_mappo_subway50 \
    --model_path ./${EXP}/cd_mappo/cd_mappo_subway50/cd_mappo_subway50_final.pth \
    --scenario_path subway_station_scenarios/subway_station_max_ue_120 \
    --save_path ./${EXP}/cd_mappo \
    --vdo_alpha 0.5 --lambda_cf 0.5 --K_actor 4 --deba_memory_size 5000 --deba_sample_size 64 --n_heads 4 --max_n 120

python test_with_action.py --agent cd_mappo --model_name cd_mappo_subway50 \
    --model_path ./${EXP}/cd_mappo/cd_mappo_subway50/cd_mappo_subway50_final.pth \
    --scenario_path subway_station_scenarios/subway_station_max_ue_150 \
    --save_path ./${EXP}/cd_mappo \
    --vdo_alpha 0.5 --lambda_cf 0.5 --K_actor 4 --deba_memory_size 5000 --deba_sample_size 64 --n_heads 4 --max_n 150

python test_with_action.py --agent cd_mappo --model_name cd_mappo_subway50 \
    --model_path ./${EXP}/cd_mappo/cd_mappo_subway50/cd_mappo_subway50_final.pth \
    --scenario_path subway_station_scenarios/subway_station_max_ue_200 \
    --save_path ./${EXP}/cd_mappo \
    --vdo_alpha 0.5 --lambda_cf 0.5 --K_actor 4 --deba_memory_size 5000 --deba_sample_size 64 --n_heads 4 --max_n 200

echo "All CD_MAPPO experiments done for seed=${SEED}."
