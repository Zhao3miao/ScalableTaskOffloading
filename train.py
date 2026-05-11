import os
import yaml
import random
import argparse
import importlib
import numpy as np
from parl.utils import logger, CSVLogger
from dynamic_env import DynamicMECEnv
from sync_vector_env import MARLSyncVectorEnv
from utils.storage import MAPPORolloutStorage
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
        "num_updates": args.num_episodes * args.ppo_epochs,
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

    if args.agent == "ippo":
        config["independent"] = True
    return config


# Training
def train(args):
    set_seed(args.seed)
    logger.info(f"Set global seed to {args.seed} for reproducibility")
    logger.info(f"Training {args.agent.upper()}")

    # Create experiment directory and CSV logger
    exp_dir = create_experiment_dir(f"{args.save_path}/{args.model_name}")
    log_file = os.path.join(exp_dir, "training_log.csv")
    csv_logger = CSVLogger(log_file)
    scenario_files = load_scenario_files(args.scenario_path, args.num_envs)
    logger.info(
        f"Loaded {len(scenario_files)} scenario files from {args.scenario_path}"
    )
    logger.info(
        "Initializing environment pool for all scenarios... (This may take a moment)"
    )

    # Create a pool of environments for all scenarios to sample from during training
    all_envs = []
    for filepath in scenario_files:
        with open(filepath, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        env = DynamicMECEnv(config=config, max_n=args.max_n)
        all_envs.append(env)
    logger.info(f"Successfully created {len(all_envs)} environments in the pool.")

    initial_envs = random.sample(all_envs, args.num_envs)
    env_fns = [lambda e=e: e for e in initial_envs]
    vector_env = MARLSyncVectorEnv(env_fns)
    logger.info(f"Vector environment created with {vector_env.num_envs} environments")

    # Build agent configuration and initialize the agent
    single_user_obs_dim = vector_env.observation_space.shape[-1]
    single_user_act_dim = vector_env.action_space.nvec[0]
    agent_module = get_agent_module(args.agent)
    AgentClass = get_agent_class(agent_module, args.agent)
    logger.info(f"Using Agent class: {AgentClass.__name__}")

    agent_config = build_agent_config(
        args, vector_env, single_user_obs_dim, single_user_act_dim
    )
    agent = AgentClass(agent_config)

    # Main training loop
    num_episodes = args.num_episodes
    for ep in range(1, num_episodes + 1):
        sampled_envs = random.sample(all_envs, args.num_envs)
        vector_env.envs = sampled_envs
        obs, infos = vector_env.reset()
        ep_rewards = 0.0
        ep_metrics = []
        n_agents = obs.shape[1]
        n_servers = (obs.shape[-1] - 6) // 3
        confounder_dim = 2 * n_servers
        needs_congestion = args.agent in ("cd_mappo", "att_mappo")
        if args.agent == "cd_mappo":
            agent.init_episode(
                env_num=vector_env.num_envs,
                n_agents=n_agents,
                max_steps=args.ppo_epochs * int(vector_env.max_time),
            )
        rollout_storage = MAPPORolloutStorage(
            step_nums=args.ppo_epochs * int(vector_env.max_time),
            env_num=vector_env.num_envs,
            obs_space=vector_env.observation_space,
            act_space=vector_env.action_space,
        )
        while True:
            if needs_congestion:
                cong_all = obs[:, :, -confounder_dim:]
                server_congestions = np.max(cong_all, axis=1)
                value, action, logprob = agent.sample(
                    obs, server_congestions=server_congestions
                )
            else:
                server_congestions = None
                value, action, logprob = agent.sample(obs)
            actions_list = [action[i] for i in range(vector_env.num_envs)]
            next_obs, rewards, terminations, truncations, infos = vector_env.step(
                actions_list
            )
            ep_rewards += rewards.sum()

            # Ensure the environment provides active masks and decision masks
            assert (
                "active_ue_masks" in infos
            ), "Environment must provide 'active_ue_masks' in info"
            active_masks = infos.get("active_ue_masks", np.ones_like(terminations))
            if isinstance(active_masks, list):
                active_masks = np.array(active_masks)
            active_masks = active_masks.reshape(vector_env.num_envs, n_agents, 1)

            decision_masks = infos.get("decision_masks", active_masks)
            if isinstance(decision_masks, list):
                decision_masks = np.array(decision_masks)
            decision_masks = decision_masks.reshape(vector_env.num_envs, n_agents, 1)

            # Reshape for storage: (env_num, n_agents, 1) to align with agent's expected input
            value = value.reshape(vector_env.num_envs, n_agents, 1)
            logprob = logprob.reshape(vector_env.num_envs, n_agents, 1)
            rewards = rewards.reshape(vector_env.num_envs, n_agents, 1)
            terminations = terminations.reshape(vector_env.num_envs, n_agents, 1)
            truncations = truncations.reshape(vector_env.num_envs, n_agents, 1)
            rollout_storage.append(
                obs=obs,
                action=action,
                logprob=logprob,
                reward=rewards,
                value=value,
                done=terminations,
                active_mask=active_masks,
                decision_mask=decision_masks,
                congestion=server_congestions,
            )
            
            obs = next_obs
            valid_terms = (terminations | truncations)[active_masks.astype(bool)]
            if np.all(valid_terms) and len(valid_terms) > 0:
                ep_metrics.extend(collect_global_state_metrics(infos))
                break
        
        # Compute advantages and update the agent
        rollout_storage.compute_gae(gamma=args.gamma, gae_lambda=args.gae_lambda)
        if args.agent == "cd_mappo":
            agent.compute_deconf_advantages(
                rollout_storage,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                current_episode=ep,
                total_episodes=num_episodes,
            )
        mean_pg, mean_v, mean_ent, _ = agent.learn(rollout_storage)
        
        avg_reward = ep_rewards / (vector_env.num_envs * vector_env.max_time)
        episode_metrics = compute_episode_metrics(ep_metrics)
        log_metrics = {
            "episode": ep,
            "avg_reward": round(avg_reward, 4),
            "avg_tasks": round(episode_metrics["avg_tasks"], 2),
            "avg_cost": round(episode_metrics["avg_cost"], 4),
            "avg_drop_ratio": round(episode_metrics["avg_drop_ratio"], 2),
            "v_loss": round(mean_v, 4),
            "p_loss": round(mean_pg, 4),
            "entropy": round(mean_ent, 4),
        }
        logger.info(format_training_log(ep, log_metrics))
        csv_logger.log_dict(log_metrics)

        # Save model at specified intervals and at the end of training
        if ep % args.save_freq == 0 or ep == num_episodes:
            if ep == num_episodes:
                model_path = os.path.join(exp_dir, f"{args.model_name}_final.pth")
            else:
                model_path = os.path.join(exp_dir, f"{args.model_name}_ep{ep}.pth")
            agent.save(model_path)
            logger.info(f"Model saved to {model_path}")
        rollout_storage.reset()
    vector_env.close()
    logger.info("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        help="Agent module name (e.g., ippo, cd_mappo, att_mappo)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--num_episodes", type=int, default=300, help="Total training episodes"
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
        "--max_n", type=int, default=100, help="Maximum number of agents"
    )
    parser.add_argument(
        "--save_freq", type=int, default=50, help="How often to save the model"
    )
    parser.add_argument(
        "--save_path", type=str, default="./experiment", help="Path to save the model"
    )
    parser.add_argument(
        "--num_envs", type=int, default=5, help="Number of parallel environments"
    )
    parser.add_argument(
        "--scenario_path",
        type=str,
        default="campus_scenarios/campus_max_ue_50",
        help="Path to scenario directory",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Name of the model (used for logging and saving). Defaults to agent name.",
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
    train(args)
