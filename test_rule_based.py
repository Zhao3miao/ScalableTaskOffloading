import argparse
import csv
import os
from collections import Counter

import numpy as np
import yaml
from parl.utils import logger

from agents.rule_based_agent import RULE_BASED_AGENT_CLASSES
from dynamic_env import DynamicMECEnv
from sync_vector_env import MARLSyncVectorEnv
from utils.utils import (
    collect_global_state_metrics,
    compute_episode_metrics,
    create_experiment_dir,
    format_training_log,
    load_scenario_files,
    set_seed,
)


def build_agent_config(args, vector_env):
    obs_dim = vector_env.observation_space.shape[-1]
    act_dim = vector_env.envs[0].action_space.nvec[0]
    return {
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "max_n": args.max_n,
        "seed": args.seed,
        "n_servers": act_dim - 1,
    }


def make_env(filepath, max_n):
    with open(filepath, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return DynamicMECEnv(config=config, max_n=max_n)


def update_result_csv(log_file, scenario_name, log_metrics):
    file_exists = os.path.isfile(log_file) and os.path.getsize(log_file) > 0
    rows = []
    header = list(log_metrics.keys())
    updated = False

    if file_exists:
        with open(log_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                for field in reader.fieldnames:
                    if field not in header:
                        header.append(field)
                for row in reader:
                    if row.get("scenario") == scenario_name:
                        new_row = row.copy()
                        new_row.update(log_metrics)
                        rows.append(new_row)
                        updated = True
                    else:
                        rows.append(row)

    if not updated:
        rows.append(log_metrics)

    with open(log_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args):
    set_seed(args.seed)
    if args.agent not in RULE_BASED_AGENT_CLASSES:
        valid = ", ".join(sorted(RULE_BASED_AGENT_CLASSES.keys()))
        raise ValueError(f"Unknown rule-based agent '{args.agent}'. Valid: {valid}")

    logger.info(f"Set global seed to {args.seed} for reproducibility")
    logger.info(f"Evaluating rule-based agent: {args.agent}")

    if not args.model_name:
        args.model_name = args.agent
    results_dir = create_experiment_dir(f"{args.save_path}/{args.model_name}")
    log_file = os.path.join(results_dir, "test_results.csv")

    scenario_files = load_scenario_files(args.scenario_path, args.num_envs)
    logger.info(
        f"Loaded {len(scenario_files)} scenario files from {args.scenario_path}"
    )

    env_fns = [
        (lambda filepath=filepath: make_env(filepath, args.max_n))
        for filepath in scenario_files
    ]
    vector_env = MARLSyncVectorEnv(env_fns)
    logger.info(f"Vector environment created with {vector_env.num_envs} environments")

    agent_config = build_agent_config(args, vector_env)
    agent = RULE_BASED_AGENT_CLASSES[args.agent](agent_config)
    action_dim = agent_config["act_dim"]

    obs, infos = vector_env.reset()
    ep_rewards = 0.0
    ep_metrics = []
    ep_steps = 0
    all_decision_actions = []

    while True:
        action = agent.predict(obs)
        decision_masks = obs[:, :, 1] > 0.5
        if np.any(decision_masks):
            all_decision_actions.extend(action[decision_masks].astype(int).tolist())

        actions_list = [action[i] for i in range(vector_env.num_envs)]
        next_obs, rewards, terminations, truncations, infos = vector_env.step(
            actions_list
        )
        ep_rewards += rewards.sum()
        ep_steps += 1
        obs = next_obs

        active_masks = infos.get("active_ue_masks", np.ones_like(terminations))
        if isinstance(active_masks, list):
            active_masks = np.array(active_masks)
        active_masks = active_masks.reshape(vector_env.num_envs, args.max_n)
        valid_terms = (terminations | truncations)[active_masks.astype(bool)]
        if np.all(valid_terms) and len(valid_terms) > 0:
            ep_metrics.extend(collect_global_state_metrics(infos))
            break

    episode_metrics = compute_episode_metrics(ep_metrics)
    avg_reward = ep_rewards / (vector_env.num_envs * vector_env.max_time)
    vector_env.close()

    action_counts = Counter(all_decision_actions)
    total_actions = sum(action_counts.values())
    scenario_name = os.path.basename(os.path.normpath(args.scenario_path))

    log_metrics = {
        "scenario": scenario_name,
        "avg_reward": round(avg_reward, 4),
        "avg_tasks": round(episode_metrics["avg_tasks"], 2),
        "avg_cost": round(episode_metrics["avg_cost"], 4),
        "avg_drop_ratio": round(episode_metrics["avg_drop_ratio"], 2),
    }
    for action_idx in range(action_dim):
        count = action_counts.get(action_idx, 0)
        ratio = (count / total_actions) * 100 if total_actions > 0 else 0.0
        log_metrics[f"action_{action_idx}_ratio"] = round(ratio, 2)

    update_result_csv(log_file, scenario_name, log_metrics)
    logger.info(format_training_log(1, log_metrics))

    dist_parts = []
    for action_idx in range(action_dim):
        pct = log_metrics[f"action_{action_idx}_ratio"]
        dist_parts.append(f"Act {action_idx}: {pct:.1f}%")

    logger.info("=== RULE-BASED EVALUATION COMPLETED ===")
    logger.info(f"Agent: {args.agent} | Scenario: {args.scenario_path}")
    logger.info(
        f"Reward: {avg_reward:.2f} | Cost: {episode_metrics['avg_cost']:.2f}s | "
        f"Drop: {episode_metrics['avg_drop_ratio']:.1f}% | "
        f"Tasks: {episode_metrics['avg_tasks']:.1f}"
    )
    logger.info(f"Decision Action Dist: {' | '.join(dist_parts)}")
    logger.info(f"Steps: {ep_steps}")
    logger.info("=======================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate rule-based MEC policies")
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        choices=sorted(RULE_BASED_AGENT_CLASSES.keys()),
        help="Rule-based policy name",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--num_envs", type=int, default=5, help="Number of parallel environments"
    )
    parser.add_argument(
        "--scenario_path",
        type=str,
        default="campus_scenarios/campus_max_ue_50",
        help="Path to test scenario directory",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Name used for result directory. Defaults to agent name.",
    )
    parser.add_argument(
        "--max_n", type=int, default=100, help="Maximum number of agents"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./experiment/rule_based",
        help="Path to save test results",
    )
    args = parser.parse_args()
    evaluate(args)
