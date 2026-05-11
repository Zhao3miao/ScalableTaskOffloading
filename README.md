# Scalable Task Offloading under Dynamic User Populations in Mobile Edge Computing via Causal Deconfounded Multi-Agent Reinforcement Learning

> **⚠️ Disclaimer**:  
> 1. This project and its corresponding paper have not yet undergone peer review.  The code is provided for research purposes only. Please do __NOT__ download or use this repo.  
> 2. The content of this repo may be modified, added, or removed depending on the review status.  
> 3. The results generated in this repo do not represent the final results presented in the paper, as some content requires manual processing. 

## 📖 Project Overview

This repository implements **CD-MAPPO**, a Causal Deconfounded Multi-Agent Reinforcement Learning method for dynamic **task Offloading** in **Mobile Edge Computing (MEC)** systems with time-varying user populations.

In realistic MEC scenarios, the number, location, and mobility pattern of user equipments (UEs) change over time. A policy trained under one user scale may fail when the test-time population increases, because server load, wireless channel quality, and task deadlines shift together. CD-MAPPO is designed to study this scalability and distribution-shift problem in multi-agent task offloading.

The main contributions are summarized as follows:

1. **Dynamic MEC Environment**: Simulates user arrival, departure, mobility, task generation, wireless transmission, edge computing, local execution, deadline misses, and disconnection-induced drops.
2. **Scalable Multi-Agent Policies**: Uses shared decentralized actors so that execution naturally supports different numbers of active UEs.
3. **Counterfactual Deconfounding**: Uses server-load interventions in the centralized critic and actor-side regularization to improve robustness under unseen load distributions.
4. **OSM-Based Scenario Generation**: Provides campus and subway-station scenarios generated from OpenStreetMap data, covering both regular mobility and event-driven crowd dynamics.



## 🖼️ Visualizations

Below are example visualizations of the two OSM-based dynamic MEC scenarios.

<p align="center">
  <strong>Campus Scenario</strong><br>
  <img src="plot/outputs/campus_ue100_scenario0.gif" width="80%" alt="Campus Scenario Animation">
</p>

<p align="center">
  <strong>Subway-Station Scenario</strong><br>
  <img src="plot/outputs/subway_ue100_scenario0.gif" width="80%" alt="Subway Station Scenario Animation">
</p>


## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Zhao3miao/ScalableTaskOffloading
   cd ScalableTaskOffloading
   ```

2. **Create and activate the Conda environment:**
   ```bash
   conda create -n scalable-task-offloading python=3.7
   conda activate scalable-task-offloading
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Quick Start & Reproduction

### 1. View Existing Results

*   **Download Pre-trained Models**: Download the trained models from [this link](https://drive.google.com/file/d/1wsAjTgPXBi1rzPCJq_kZiZV5ijG4ZSJA/view?usp=sharing) (approximately 1GB).
* **Aggregated Results**: Run `python aggregate_test_results.py` after evaluation to generate `aggregated_test_results.csv` and `aggregated_test_display.csv`.
* **Analysis Figures**: Use scripts under `plot/` to draw high-load action distributions and subway event-response curves.


### 2. Reproduce Experiments

Follow these steps to regenerate scenarios, train policies, evaluate them, and aggregate the results.

#### Step 1: Scenario Generation (Optional)

> **Note**: This step involves randomness. The repository already contains pre-generated scenario YAML files. Regenerating scenarios involves randomness and may lead to slightly different numerical results. To reproduce reported results exactly, use the provided scenario folders.

```bash
# Generate campus scenarios
bash campus_scenarios/generate_scenarios.sh 

# Generate subway-station scenarios
bash subway_station_scenarios/generate_scenarios.sh
```

#### Step 2: Training and Evaluation

Run a single method on one scenario:

```bash
# CD-MAPPO on the campus scenario
bash campus_scenarios/cd_mappo.sh 42

# CD-MAPPO on the subway-station scenario
bash subway_station_scenarios/cd_mappo.sh 42
```

Run the full experiment suite with three random seeds:
> **Note**: We recommend executing the commands in each script in parallel to speed up the training process.

```bash

bash experiment1.sh   # seed 42
bash experiment2.sh   # seed 233
bash experiment3.sh   # seed 666
```

These scripts train and evaluate:

* **Rule-based baselines**: local-only, random, best-channel, and least-congested.
* **Learning-based baselines**: Blind-IPPO, IPPO, and ATT-MAPPO.
* **Proposed method**: CD-MAPPO.
* **Ablation and sensitivity variants**: actor-loss ablation, deconfounded-critic ablation, and `vdo_alpha` sensitivity settings.


#### Step 3: Aggregate Results

Collect all evaluation results into summary CSV files:

```bash
python aggregate_test_results.py
```

This produces:

```text
aggregated_test_results.csv   # numeric metrics and confidence intervals
aggregated_test_display.csv   # display-friendly summary
```

#### Step 4: Generate Figures

Example commands:

```bash
# Visualize a campus scenario
python plot/visualize_campus_scenario.py \
  --scenario campus_scenarios/campus_max_ue_100/scenario_0.yaml \
  --save plot/outputs/campus_ue100_scenario0.gif

# Visualize a subway-station scenario
python plot/visualize_subway_scenario.py \
  --scenario subway_station_scenarios/subway_station_max_ue_100/scenario_0.yaml \
  --osm_file subway_station_scenarios/subway_station.osm \
  --save plot/outputs/subway_ue100_scenario0.gif

# Plot high-load action distributions
python plot/plot_high_load_action_distribution.py
```

## 📄License

This project uses multiple open-source components.

* **Code**: Released under the license specified in [LICENSE](LICENSE).
* **Data**: Scenario data is generated from OpenStreetMap and is subject to ODbL terms.
* **Dependencies**: Third-party Python packages are distributed under their own licenses.


## 🔗 Citation

If you use this project in your research, please cite:

```bibtex
TODO
```
