#!/bin/bash
set -e

# Run all CD-MAPPO ablation studies.
# Default seeds: 42, 233, 666.
# Usage:
#   bash ablation_study/experiment_ablation_study.sh
#   bash ablation_study/experiment_ablation_study.sh 42
#   bash ablation_study/experiment_ablation_study.sh 42 233 666

bash ablation_study/experiment_wo_actor_loss.sh "$@"
bash ablation_study/experiment_wo_deconf_critic.sh "$@"
