bash campus_scenarios/local_only.sh 42
bash campus_scenarios/random.sh 42
bash campus_scenarios/best_channel.sh 42
bash campus_scenarios/least_congested.sh 42
bash campus_scenarios/blind_ippo.sh 42
bash campus_scenarios/ippo.sh 42
bash campus_scenarios/att_mappo.sh 42
bash campus_scenarios/cd_mappo.sh 42

# CD-MAPPO parameter sensitivity on campus scenario.
# vdo_alpha=0.7 is covered by the main campus cd_mappo run above.
bash campus_scenarios/cd_mappo_vdo_alpha_01.sh 42
bash campus_scenarios/cd_mappo_vdo_alpha_03.sh 42
bash campus_scenarios/cd_mappo_vdo_alpha_05.sh 42

bash subway_station_scenarios/local_only.sh 42
bash subway_station_scenarios/random.sh 42
bash subway_station_scenarios/best_channel.sh 42
bash subway_station_scenarios/least_congested.sh 42
bash subway_station_scenarios/blind_ippo.sh 42
bash subway_station_scenarios/ippo.sh 42
bash subway_station_scenarios/att_mappo.sh 42
bash subway_station_scenarios/cd_mappo.sh 42

# CD-MAPPO ablation studies on both scenarios.
bash ablation_study/experiment_ablation_study.sh 42

# CD-MAPPO parameter sensitivity on subway station scenario.
# vdo_alpha=0.5 is covered by the main cd_mappo run above.
bash subway_station_scenarios/cd_mappo_vdo_alpha_01.sh 42
bash subway_station_scenarios/cd_mappo_vdo_alpha_03.sh 42
bash subway_station_scenarios/cd_mappo_vdo_alpha_07.sh 42
