import os
import yaml
import argparse
import importlib
import numpy as np
from collections import Counter
from parl.utils import logger
from dynamic_env import DynamicMECEnv
from sync_vector_env import MARLSyncVectorEnv
from utils.utils import (
    set_seed,
    load_scenario_files,
    create_experiment_dir,
    collect_global_state_metrics,
    compute_episode_metrics,
    format_training_log,
)


# Dynamically import agent modules and classes
def get_agent_module(agent_name):
    module_name_map = {
        "ippo": "ppo_agent",
        "cd_mappo": "cd_mappo_agent",
        "att_mappo": "att_mappo_agent",
        "blind_ippo": "blind_ippo_agent",
    }
    module_name = module_name_map.get(agent_name, agent_name)
    try:
        return importlib.import_module(f"agents.{module_name}")
    except ImportError as e:
        raise ImportError(f"Cannot import agent module 'agents.{module_name}': {e}")


# Dynamically find the Agent class in the module
def get_agent_class(agent_module, agent_name):
    class_name_map = {
        "ippo": "PPOAgent",
        "cd_mappo": "CDMAPPOAgent",
        "att_mappo": "AttMAPPOAgent",
        "blind_ippo": "BlindIPPOAgent",
    }
    class_name = class_name_map.get(agent_name)
    if class_name is None:
        for attr_name in dir(agent_module):
            if "Agent" in attr_name and not attr_name.startswith("_"):
                return getattr(agent_module, attr_name)
        raise AttributeError(
            f"Cannot find Agent class in module 'agents.{agent_module.__name__}'"
        )
    return getattr(agent_module, class_name)


# Build a configuration dictionary for the agent based on command-line arguments and environment details
def build_agent_config(args, vector_env, single_user_obs_dim, single_user_act_dim):
    config = {
        "obs_dim": single_user_obs_dim,
        "act_dim": single_user_act_dim,
        "max_n": args.max_n,
        "hidden_size_actor": args.hidden_size_actor,
        "hidden_size_critic": args.hidden_size_critic,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_param": args.clip_param,
        "entropy_coef": args.entropy_coef,
        "value_coef": args.value_coef,
        "initial_lr": args.lr,
        "lr_decay": args.lr_decay,
        "num_updates": 0,
        "update_epochs": args.ppo_epochs,
        "num_minibatches": args.num_minibatches,
        "batch_size": args.batch_size,
        "max_grad_norm": args.max_grad_norm,
    }
    if args.agent == "cd_mappo":
        config["n_servers"] = (single_user_obs_dim - 6) // 3
        config["confounder_dim"] = 2 * config["n_servers"]
        config["n_heads"] = args.n_heads
        config["vdo_alpha"] = args.vdo_alpha
        config["lambda_cf"] = args.lambda_cf
        config["K_actor"] = args.K_actor
        config["deba_memory_size"] = args.deba_memory_size
        config["deba_sample_size"] = args.deba_sample_size
    if args.agent == "att_mappo":
        config["n_servers"] = (single_user_obs_dim - 6) // 3
        config["confounder_dim"] = 2 * config["n_servers"]
        config["n_heads"] = args.n_heads
    return config


# Evaluation
def evaluate(args):
    set_seed(args.seed)
    logger.info(f"Set global seed to {args.seed} for reproducibility")
    logger.info(f"Using {args.num_envs} parallel environments")
    logger.info(f"Evaluating {args.agent.upper()}")

    # Create experiment directory and CSV logger
    if not args.model_name:
        args.model_name = args.agent
    results_dir = create_experiment_dir(f"{args.save_path}/{args.model_name}")
    log_file = os.path.join(results_dir, "test_results.csv")
    scenario_files = load_scenario_files(args.scenario_path, args.num_envs)
    logger.info(
        f"Loaded {len(scenario_files)} scenario files from {args.scenario_path}"
    )
    logger.info("Initializing environments for testing...")

    # Create a pool of environments for all scenarios to sample from during testing
    all_envs = []
    for filepath in scenario_files:
        with open(filepath, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        env = DynamicMECEnv(config=config, max_n=args.max_n)
        all_envs.append(env)
    env_fns = [lambda e=e: e for e in all_envs]
    vector_env = MARLSyncVectorEnv(env_fns)
    logger.info(f"Vector environment created with {vector_env.num_envs} environments")

    # Build agent configuration and initialize the agent
    single_user_obs_dim = vector_env.observation_space.shape[-1]
    single_user_act_dim = vector_env.envs[0].action_space.nvec[0]
    agent_module = get_agent_module(args.agent)
    AgentClass = get_agent_class(agent_module, args.agent)
    logger.info(f"Using Agent class: {AgentClass.__name__}")
    agent_config = build_agent_config(
        args, vector_env, single_user_obs_dim, single_user_act_dim
    )
    agent = AgentClass(agent_config)

    # Load model weights
    if args.model_path:
        if os.path.exists(args.model_path):
            agent.restore(args.model_path)
            logger.info(f"Model loaded from {args.model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {args.model_path}")
    else:
        logger.warning("No model path specified, using untrained model")
    obs, infos = vector_env.reset()
    ep_rewards = 0.0
    ep_metrics = []
    ep_steps = 0
    all_actions = []
    n_agents = obs.shape[1]

    # Main evaluation loop
    while True:
        action = agent.predict(obs)
        actions_list = [action[i] for i in range(vector_env.num_envs)]

        # Ensure the environment provides active masks
        active_masks = infos.get(
            "active_ue_masks", np.ones((vector_env.num_envs, n_agents))
        )
        if isinstance(active_masks, list):
            active_masks = np.array(active_masks)
        active_masks = active_masks.reshape(vector_env.num_envs, n_agents)
        for env_idx in range(vector_env.num_envs):
            active_mask = active_masks[env_idx]
            for agent_idx in range(n_agents):
                if active_mask[agent_idx] == 1:
                    all_actions.append(action[env_idx][agent_idx])
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
    action_counts = Counter(np.array(all_actions).flatten())
    total_actions = sum(action_counts.values())
    if args.scenario_path:
        scenario_name = os.path.basename(os.path.normpath(args.scenario_path))
    else:
        scenario_name = "default_scenario"
    log_metrics = {
        "scenario": scenario_name,
        "avg_reward": round(avg_reward, 4),
        "avg_tasks": round(episode_metrics["avg_tasks"], 2),
        "avg_cost": round(episode_metrics["avg_cost"], 4),
        "avg_drop_ratio": round(episode_metrics["avg_drop_ratio"], 2),
    }
    for i in range(single_user_act_dim):
        count = action_counts.get(i, 0)
        percentage = (count / total_actions) * 100 if total_actions > 0 else 0
        log_metrics[f"action_{i}_ratio"] = round(percentage, 2)
    logger.info(format_training_log(1, log_metrics))
    import csv

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
    
    dist_strs = []
    for act in sorted(action_counts.keys()):
        percentage = (action_counts[act] / total_actions) * 100 if total_actions > 0 else 0
        dist_strs.append(f"Act {act}: {percentage:.1f}%")
    dist_str = " | ".join(dist_strs)

    logger.info("=== EVALUATION COMPLETED ===")
    logger.info(f"Agent: {args.agent.upper()} | Scenario: {args.scenario_path}")
    logger.info(f"Reward: {avg_reward:.2f} | Cost: {episode_metrics['avg_cost']:.2f}s | Drop: {episode_metrics['avg_drop_ratio']:.1f}% | Tasks: {episode_metrics['avg_tasks']:.1f}")
    logger.info(f"Action Dist: {dist_str}")
    logger.info("============================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test action distribution script")
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        help="Agent module name (e.g., ippo, cd_mappo, att_mappo)",
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
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model file (.pth)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Name of the model (used for logging). Defaults to agent name.",
    )
    parser.add_argument(
        "--max_n", type=int, default=100, help="Maximum number of agents"
    )
    parser.add_argument(
        "--ppo_epochs", type=int, default=4, help="PPO update epochs per episode"
    )
    parser.add_argument(
        "--num_minibatches", type=int, default=4, help="Number of minibatches"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for PPO updates"
    )
    parser.add_argument(
        "--hidden_size_actor", type=int, default=128, help="Actor hidden layer size"
    )
    parser.add_argument(
        "--hidden_size_critic", type=int, default=256, help="Critic hidden layer size"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--lr_decay", action="store_true", help="Use learning rate decay"
    )
    parser.add_argument(
        "--clip_param", type=float, default=0.2, help="PPO clip parameter"
    )
    parser.add_argument(
        "--entropy_coef", type=float, default=0.01, help="Entropy coefficient"
    )
    parser.add_argument(
        "--value_coef", type=float, default=0.5, help="Value loss coefficient"
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--gae_lambda", type=float, default=0.95, help="GAE lambda parameter"
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=0.5, help="Max gradient norm"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./experiment",
        help="Path to save test results",
    )
    parser.add_argument(
        "--vdo_alpha",
        type=float,
        default=0.7,
        help="V_mix = α*V_do + (1-α)*V_factual - Only for CD-MAPPO",
    )
    parser.add_argument(
        "--lambda_cf",
        type=float,
        default=0.5,
        help="Actor counterfactual loss weight - Only for CD-MAPPO",
    )
    parser.add_argument(
        "--K_actor",
        type=int,
        default=4,
        help="Actor counterfactual sampling number - Only for CD-MAPPO",
    )
    parser.add_argument(
        "--deba_memory_size",
        type=int,
        default=5000,
        help="DEBA experience buffer size - Only for CD-MAPPO",
    )
    parser.add_argument(
        "--deba_sample_size",
        type=int,
        default=64,
        help="V_do sample size per update - Only for CD-MAPPO",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=4,
        help="Attention Critic head count - Only for CD-MAPPO / ATT-MAPPO",
    )
    args = parser.parse_args()
    if args.model_name is None:
        args.model_name = args.agent
    evaluate(args)
