from train_cd_mappo_ablation_utils import run_cd_mappo_ablation


def configure_args(args):
    pass


if __name__ == "__main__":
    run_cd_mappo_ablation(
        description="Train CD-MAPPO without the deconfounded critic.",
        default_model_name="cd_mappo_wo_deconf_critic",
        configure_args=configure_args,
        extra_config={"use_deconf_critic": False},
    )
