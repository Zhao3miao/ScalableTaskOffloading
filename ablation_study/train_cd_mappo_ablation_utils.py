import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import train as train_module


def build_parser(description):
    parser = argparse.ArgumentParser(description=description)
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
        help="Name of the model (used for logging and saving).",
    )
    parser.add_argument(
        "--vdo_alpha",
        type=float,
        default=0.7,
        help="V_mix = alpha*V_do + (1-alpha)*V_factual.",
    )
    parser.add_argument(
        "--lambda_cf",
        type=float,
        default=0.5,
        help="Actor counterfactual loss weight.",
    )
    parser.add_argument(
        "--K_actor",
        type=int,
        default=4,
        help="Actor counterfactual sampling number.",
    )
    parser.add_argument(
        "--deba_memory_size",
        type=int,
        default=5000,
        help="DEBA experience buffer size.",
    )
    parser.add_argument(
        "--deba_sample_size",
        type=int,
        default=64,
        help="V_do sample size per update.",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=4,
        help="Attention critic head count.",
    )
    parser.add_argument(
        "--fl_sync_freq",
        type=int,
        default=10,
        help="Kept for compatibility with train.train.",
    )
    parser.add_argument(
        "--fl_local_epochs",
        type=int,
        default=1,
        help="Kept for compatibility with train.build_agent_config.",
    )
    parser.add_argument(
        "--fl_critic_epochs",
        type=int,
        default=1,
        help="Kept for compatibility with train.build_agent_config.",
    )
    parser.add_argument(
        "--fl_max_clients",
        type=int,
        default=20,
        help="Kept for compatibility with train.build_agent_config.",
    )
    parser.add_argument(
        "--fl_min_decision_samples",
        type=int,
        default=4,
        help="Kept for compatibility with train.build_agent_config.",
    )
    parser.add_argument(
        "--fl_server_mix_beta",
        type=float,
        default=0.35,
        help="Kept for compatibility with train.build_agent_config.",
    )
    parser.add_argument(
        "--fl_prox_mu",
        type=float,
        default=1e-3,
        help="Kept for compatibility with train.build_agent_config.",
    )
    return parser


def run_cd_mappo_ablation(description, default_model_name, configure_args, extra_config):
    parser = build_parser(description)
    args = parser.parse_args()
    args.agent = "cd_mappo"
    if args.model_name is None:
        args.model_name = default_model_name
    configure_args(args)

    original_build_agent_config = train_module.build_agent_config

    def build_agent_config_with_ablation(
        args, vector_env, single_user_obs_dim, single_user_act_dim
    ):
        config = original_build_agent_config(
            args, vector_env, single_user_obs_dim, single_user_act_dim
        )
        config.update(extra_config)
        return config

    train_module.build_agent_config = build_agent_config_with_ablation
    try:
        train_module.train(args)
    finally:
        train_module.build_agent_config = original_build_agent_config
