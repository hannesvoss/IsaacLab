# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class Ur5RLPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 128
    max_iterations = 1500
    save_interval = 25
    experiment_name = "ur5_rl_direct"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.6,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=8,
        num_mini_batches=8,
        learning_rate=2.5e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.5,
    )
