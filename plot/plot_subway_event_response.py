import argparse
import importlib
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dynamic_env import DynamicMECEnv
from sync_vector_env import MARLSyncVectorEnv
from utils.utils import load_scenario_files, set_seed


AGENT_MODULES = {
    "ippo": "ppo_agent",
    "cd_mappo": "cd_mappo_agent",
    "att_mappo": "att_mappo_agent",
    "blind_ippo": "blind_ippo_agent",
}

AGENT_CLASSES = {
    "ippo": "PPOAgent",
    "cd_mappo": "CDMAPPOAgent",
    "att_mappo": "AttMAPPOAgent",
    "blind_ippo": "BlindIPPOAgent",
}


def get_agent_class(agent_name):
    module_name = AGENT_MODULES.get(agent_name, agent_name)
    module = importlib.import_module(f"agents.{module_name}")
    class_name = AGENT_CLASSES.get(agent_name)
    if class_name:
        return getattr(module, class_name)
    for attr_name in dir(module):
        if "Agent" in attr_name and not attr_name.startswith("_"):
            return getattr(module, attr_name)
    raise AttributeError(f"Cannot find agent class for {agent_name}")


def build_agent_config(args, agent_name, obs_dim, act_dim):
    config = {
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "max_n": args.max_n,
        "hidden_size_actor": args.hidden_size_actor,
        "hidden_size_critic": args.hidden_size_critic,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_param": args.clip_param,
        "entropy_coef": args.entropy_coef,
        "value_coef": args.value_coef,
        "initial_lr": args.lr,
        "lr_decay": False,
        "num_updates": 0,
        "update_epochs": args.ppo_epochs,
        "num_minibatches": args.num_minibatches,
        "batch_size": args.batch_size,
        "max_grad_norm": args.max_grad_norm,
    }
    if agent_name in {"cd_mappo", "att_mappo"}:
        n_servers = (obs_dim - 6) // 3
        config["n_servers"] = n_servers
        config["confounder_dim"] = 2 * n_servers
        config["n_heads"] = args.n_heads
    if agent_name == "cd_mappo":
        config["vdo_alpha"] = args.vdo_alpha
        config["lambda_cf"] = args.lambda_cf
        config["K_actor"] = args.K_actor
        config["deba_memory_size"] = args.deba_memory_size
        config["deba_sample_size"] = args.deba_sample_size
    return config


def make_vector_env(args):
    scenario_files = load_scenario_files(args.scenario_path, args.num_envs)[: args.num_envs]
    envs = []
    for filepath in scenario_files:
        with open(filepath, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        envs.append(DynamicMECEnv(config=config, max_n=args.max_n))
    return MARLSyncVectorEnv([lambda e=e: e for e in envs])


def as_mask_array(mask_like, num_envs, n_agents):
    if mask_like is None:
        return np.zeros((num_envs, n_agents), dtype=bool)
    arr = np.asarray(mask_like)
    return arr.reshape(num_envs, n_agents).astype(bool)


def collect_agent_trace(args, label, agent_name, model_path):
    vector_env = make_vector_env(args)
    obs_dim = vector_env.observation_space.shape[-1]
    act_dim = vector_env.envs[0].action_space.nvec[0]
    n_agents = vector_env.envs[0].n_agents

    AgentClass = get_agent_class(agent_name)
    agent = AgentClass(build_agent_config(args, agent_name, obs_dim, act_dim))
    agent.restore(model_path)

    obs, infos = vector_env.reset()
    rows = []
    prev_completed = [env.completed_tasks for env in vector_env.envs]
    prev_dropped = [
        env.drop_count_disconnection + env.drop_count_deadline
        for env in vector_env.envs
    ]

    while True:
        actions = agent.predict(obs)
        actions_list = [actions[i] for i in range(vector_env.num_envs)]
        obs, rewards, terminations, truncations, infos = vector_env.step(actions_list)

        active_masks = as_mask_array(
            infos.get("active_ue_masks"), vector_env.num_envs, n_agents
        )
        decision_masks = as_mask_array(
            infos.get("decision_masks"), vector_env.num_envs, n_agents
        )

        active_count = 0
        decision_count = 0
        action_counts = np.zeros(act_dim, dtype=int)
        new_completed = 0
        new_dropped = 0

        for env_idx, env in enumerate(vector_env.envs):
            active_count += int(active_masks[env_idx].sum())
            decision_indices = np.where(decision_masks[env_idx])[0]
            decision_count += int(len(decision_indices))
            if len(decision_indices) > 0:
                env_actions = actions[env_idx][decision_indices].astype(int)
                action_counts += np.bincount(env_actions, minlength=act_dim)[:act_dim]

            current_completed = env.completed_tasks
            current_dropped = env.drop_count_disconnection + env.drop_count_deadline
            new_completed += max(0, current_completed - prev_completed[env_idx])
            new_dropped += max(0, current_dropped - prev_dropped[env_idx])
            prev_completed[env_idx] = current_completed
            prev_dropped[env_idx] = current_dropped

        row = {
            "method": label,
            "time": float(vector_env.envs[0].current_time),
            "active_ues": active_count / vector_env.num_envs,
            "decision_tasks": decision_count / vector_env.num_envs,
            "local_ratio": np.nan
            if decision_count == 0
            else action_counts[0] / decision_count,
            "step_reward": float(np.sum(rewards)) / vector_env.num_envs,
            "new_completed": new_completed / vector_env.num_envs,
            "new_dropped": new_dropped / vector_env.num_envs,
            "new_finished_or_dropped": (new_completed + new_dropped)
            / vector_env.num_envs,
        }
        for action_idx, action_count in enumerate(action_counts):
            row[f"action_{action_idx}_count"] = action_count / vector_env.num_envs
            row[f"action_{action_idx}_ratio"] = (
                np.nan if decision_count == 0 else action_count / decision_count
            )
        rows.append(row)

        next_active_masks = as_mask_array(
            infos.get("active_ue_masks"), vector_env.num_envs, n_agents
        )
        valid_done = (terminations | truncations)[next_active_masks]
        if len(valid_done) > 0 and np.all(valid_done):
            break
        if vector_env.envs[0].current_time >= vector_env.max_time:
            break

    vector_env.close()
    return pd.DataFrame(rows)


def smooth_series(series, window):
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1, center=True).mean()


def plot_event_response(trace_df, args):
    os.makedirs(os.path.dirname(args.output_png), exist_ok=True)
    trace_df = trace_df.copy()
    trace_df["window_drop_ratio"] = np.nan
    for method in args.method_labels:
        mask = trace_df["method"] == method
        method_df = trace_df.loc[mask].sort_values("time")
        dropped = method_df["new_dropped"].rolling(
            window=args.drop_window, min_periods=1, center=True
        ).sum()
        finished = method_df["new_finished_or_dropped"].rolling(
            window=args.drop_window, min_periods=1, center=True
        ).sum()
        drop_ratio = (dropped / finished.replace(0, np.nan)) * 100.0
        trace_df.loc[method_df.index, "window_drop_ratio"] = drop_ratio

    colors = {
        "CD-MAPPO": "#1f77b4",
        "ATT-MAPPO": "#ff7f0e",
        "IPPO": "#2ca02c",
    }

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(9.0, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1]},
    )

    first_method = args.method_labels[0]
    ue_df = trace_df[trace_df["method"] == first_method].sort_values("time")
    axes[0].plot(
        ue_df["time"],
        smooth_series(ue_df["active_ues"], args.smooth_window),
        color="#333333",
        linewidth=2.0,
    )
    axes[0].set_ylabel("Active UEs")

    for method in args.method_labels:
        sub = trace_df[trace_df["method"] == method].sort_values("time")
        color = colors.get(method)
        axes[1].plot(
            sub["time"],
            smooth_series(sub["step_reward"], args.smooth_window),
            label=method,
            linewidth=2.2 if method == "CD-MAPPO" else 1.7,
            color=color,
        )
        axes[2].plot(
            sub["time"],
            smooth_series(sub["window_drop_ratio"], args.smooth_window),
            label=method,
            linewidth=2.2 if method == "CD-MAPPO" else 1.7,
            color=color,
        )

    axes[1].set_ylabel("Step reward")
    axes[2].set_ylabel("Drop ratio (%)")
    axes[2].set_xlabel("Simulation time (s)")

    for ax in axes:
        ax.axvline(args.arrival_time, color="#555555", linestyle="--", linewidth=1.1)
        ax.axvline(args.departure_time, color="#555555", linestyle="--", linewidth=1.1)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].text(
        args.arrival_time,
        axes[0].get_ylim()[1],
        " arrival",
        va="top",
        ha="left",
        fontsize=9,
        color="#555555",
    )
    axes[0].text(
        args.departure_time,
        axes[0].get_ylim()[1],
        " departure",
        va="top",
        ha="left",
        fontsize=9,
        color="#555555",
    )

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=3,
        frameon=False,
        columnspacing=1.4,
        handlelength=2.2,
    )
    fig.suptitle(
        "Event-wise Response in the Subway-station Scenario",
        y=0.995,
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(args.output_png, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot event-wise policy response in the subway-station scenario."
    )
    parser.add_argument(
        "--scenario_path",
        default="subway_station_scenarios/subway_station_max_ue_200",
    )
    parser.add_argument("--num_envs", type=int, default=5)
    parser.add_argument("--max_n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arrival_time", type=float, default=60.0)
    parser.add_argument("--departure_time", type=float, default=240.0)
    parser.add_argument("--smooth_window", type=int, default=9)
    parser.add_argument("--drop_window", type=int, default=15)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--output_csv",
        default="plot/outputs/subway_event_response_ue200_trace.csv",
    )
    parser.add_argument(
        "--output_png",
        default="plot/outputs/subway_event_response_ue200.png",
    )

    parser.add_argument("--cd_mappo_path", required=True)
    parser.add_argument("--att_mappo_path", required=True)
    parser.add_argument("--ippo_path", required=True)

    parser.add_argument("--hidden_size_actor", type=int, default=128)
    parser.add_argument("--hidden_size_critic", type=int, default=256)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--num_minibatches", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip_param", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)

    parser.add_argument("--vdo_alpha", type=float, default=0.5)
    parser.add_argument("--lambda_cf", type=float, default=0.5)
    parser.add_argument("--K_actor", type=int, default=4)
    parser.add_argument("--deba_memory_size", type=int, default=5000)
    parser.add_argument("--deba_sample_size", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)

    args = parser.parse_args()
    args.method_labels = ["CD-MAPPO", "ATT-MAPPO", "IPPO"]
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    traces = []
    specs = [
        ("CD-MAPPO", "cd_mappo", args.cd_mappo_path),
        ("ATT-MAPPO", "att_mappo", args.att_mappo_path),
        ("IPPO", "ippo", args.ippo_path),
    ]
    for label, agent_name, model_path in specs:
        print(f"Collecting trajectory for {label}: {model_path}")
        traces.append(collect_agent_trace(args, label, agent_name, model_path))

    trace_df = pd.concat(traces, ignore_index=True)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    trace_df.to_csv(args.output_csv, index=False)
    plot_event_response(trace_df, args)
    print(f"Saved trace CSV: {args.output_csv}")
    print(f"Saved figure: {args.output_png}")


if __name__ == "__main__":
    main()
