import sys
import argparse
import json
import numpy as np

from experiments.base.parser import addparse
from experiments.base.print import print_info


def run_cli(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Train DQN on Acrobot.")
    addparse(parser)
    args = parser.parse_args(argvs)
    print_info(args.experiment_name, "DQN", "Acrobot", args.bellman_iterations_scope, args.seed)
    p = json.load(
        open(f"experiments/acrobot/figures/{args.experiment_name}/parameters.json")
    )  # p for parameters

    from experiments.acrobot.utils import generate_keys
    from dqn.environments.double_pendulum import AcrobotEnv
    from dqn.environments import mpar, dt, integrator, x0, goal
    from dqn.sample_collection.replay_buffer import ReplayBuffer
    from dqn.networks.q_architectures import DoublePendulumDQN
    from experiments.base.DQN import train

    q_key, train_key = generate_keys(args.seed)

    env = AcrobotEnv(mpar, dt, integrator, x0, goal, 200)

    replay_buffer = ReplayBuffer(
        p["replay_buffer_size"],
        p["batch_size"],
        (4,),
        np.float32,
        lambda x: x,
    )

    q = DoublePendulumDQN(
        (4,),
        env.n_actions,
        p["gamma"],
        q_key,
        p["dqn_learning_rate"],
        p["n_training_steps_per_online_update"],
        p["dqn_n_training_steps_per_target_update"],
    )

    train(train_key, "acrobot", args, p, q, env, replay_buffer)
