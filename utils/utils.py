import os
import random
import glob
import csv
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch

GLOBE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def to_tensor(item, dtype=torch.float32):
    if isinstance(item, torch.Tensor):
        return item.to(GLOBE_DEVICE, dtype=dtype)
    return torch.tensor(item, dtype=dtype, device=GLOBE_DEVICE)


def load_scenario_files(scenario_path: str, num_envs: int) -> List[str]:
    files = []
    paths = scenario_path.split(',')
    for p in paths:
        p = p.strip()
        if not p:
            continue
        p = os.path.normpath(p)
        found = glob.glob(os.path.join(p, "*.yaml"))
        if not found:
            found = glob.glob(os.path.join(p, "**/*.yaml"), recursive=True)
        files.extend(found)
        
    files = list(set(files))
    files.sort()

    if not files:
        raise FileNotFoundError(f"No scenario files found in {scenario_path}")

    if len(files) < num_envs:
        print(f"Warning: Requested {num_envs} environments but only found {len(files)} files. Using all available.")

    return files


def get_scenario_file(search_path: str = "generated_scenarios/*/*.yaml") -> str:
    files = glob.glob(search_path)
    if not files:
        raise FileNotFoundError(f"No scenario files found in {search_path}")
    return files[0]



def create_experiment_dir(dir_name: str, exist_ok: bool = True) -> str:
    os.makedirs(dir_name, exist_ok=exist_ok)
    return dir_name


def get_experiment_dir(exp_type: str) -> str:
    return f"experiment/{exp_type}"




def create_csv_log(csv_file: str, headers: List[str]) -> None:
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)


def save_metrics_to_csv(csv_file: str, metrics: Dict[str, Any]) -> None:
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(metrics.values()))


def format_training_log(ep: int, metrics: Dict[str, Any]) -> str:
    return (
        f"Ep {ep:03d} | "
        f"AvgReward: {metrics.get('avg_reward', 0):.2f} | "
        f"AvgTasks: {metrics.get('avg_tasks', 0):.1f} | "
        f"AvgCost: {metrics.get('avg_cost', 0):.2f}s | "
        f"AvgDrop %: {metrics.get('avg_drop_ratio', 0):.1f}% | "
        f"vLoss: {metrics.get('v_loss', 0):.3f} | "
        f"pLoss: {metrics.get('p_loss', 0):.3f} | "
        f"Ent: {metrics.get('entropy', 0):.3f}"
    )


def collect_global_state_metrics(infos: Dict) -> List[Dict[str, Any]]:
    metrics = []

    if isinstance(infos, dict) and "final_info" in infos:
        final_info_array = infos["final_info"]
        if isinstance(final_info_array, np.ndarray):
            for item in final_info_array:
                if isinstance(item, dict):
                    if "global_state" in item:
                        metrics.append(item["global_state"])
                    elif "_final_global_state" in item:
                        metrics.append(item["_final_global_state"])

    if not metrics and isinstance(infos, dict) and "global_state" in infos:
        global_states = infos["global_state"]
        if isinstance(global_states, list):
            for gs in global_states:
                if isinstance(gs, dict):
                    metrics.append(gs)
        elif isinstance(global_states, dict):
            metrics.append(global_states)

    return metrics


def compute_episode_metrics(ep_metrics: List[Dict[str, Any]]) -> Dict[str, float]:
    if not ep_metrics:
        return {
            "avg_tasks": 0,
            "avg_cost": 0.0,
            "avg_drop_ratio": 0.0,
        }

    return {
        "avg_tasks": np.mean([m.get("total_tasks", 0) for m in ep_metrics]),
        "avg_cost": np.mean([m.get("avg_cost", 0.0) for m in ep_metrics]),
        "avg_drop_ratio": np.mean([m.get("drop_rate", 0.0) for m in ep_metrics]),
    }
