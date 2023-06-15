import sys
import argparse
import json
import numpy as np

from experiments.base.parser import addparse
from experiments.base.print import print_info


def run_cli(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Train DQN on Pendubot.")
    addparse(parser)
    args = parser.parse_args(argvs)
    print_info(args.experiment_name, "DQN", "Pendubot", args.seed)
    p = json.load(open(f"experiments/pendubot/figures/{args.experiment_name}/parameters.json"))  # p for parameters

    from experiments.pendubot.utils import generate_keys
    from dqn.environments.double_pendulum import PendubotEnv
    from dqn.environments import mpar, dt, integrator, x0, goal
    from dqn.sample_collection.replay_buffer import ReplayBuffer
    from dqn.networks.q_architectures import DoublePendulumDQN
    from experiments.base.DQN import train

    q_key, train_key = generate_keys(args.seed)

    env = PendubotEnv(mpar, dt, p["n_repeated_actions"], integrator, x0, goal, p["n_actions"])

    replay_buffer = ReplayBuffer(
        p["replay_buffer_size"],
        p["batch_size"],
        (6,),
        np.float32,
        lambda x: x,
    )

    q = DoublePendulumDQN(
        (6,),
        env.n_actions,
        p["gamma"],
        p["layers_dimensions"],
        q_key,
        p["dqn_learning_rate"],
        p["n_training_steps_per_online_update"],
        p["dqn_n_training_steps_per_target_update"],
    )

    train(train_key, "pendubot", args, p, q, env, replay_buffer)
