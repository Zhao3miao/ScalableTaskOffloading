from train_cd_mappo_ablation_utils import run_cd_mappo_ablation


def configure_args(args):
    args.lambda_cf = 0.0
    args.K_actor = 0


if __name__ == "__main__":
    run_cd_mappo_ablation(
        description="Train CD-MAPPO without the counterfactual actor loss.",
        default_model_name="cd_mappo_wo_actor_loss",
        configure_args=configure_args,
        extra_config={"lambda_cf": 0.0, "K_actor": 0},
    )
