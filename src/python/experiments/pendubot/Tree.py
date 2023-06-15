import sys
import argparse
import json
import numpy as np

from experiments.base.parser import addparse
from experiments.base.print import print_info


def run_cli(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Train Tree on Pendubot.")
    addparse(parser)
    args = parser.parse_args(argvs)
    print_info(args.experiment_name, "Tree", "Pendubot", args.seed)
    p = json.load(open(f"experiments/pendubot/figures/{args.experiment_name}/parameters.json"))  # p for parameters

    from experiments.pendubot.utils import generate_keys
    from dqn.environments.double_pendulum import PendubotEnv
    from dqn.environments import mpar, dt, integrator, x0, goal
    from dqn.sample_collection.replay_buffer import ReplayBuffer
    from experiments.base.Tree import train

    _, train_key = generate_keys(args.seed)

    env = PendubotEnv(mpar, dt, p["n_repeated_actions"], integrator, x0, goal, p["n_actions"])

    replay_buffer = ReplayBuffer(
        p["replay_buffer_size"],
        p["batch_size"],
        (6,),
        np.float32,
        lambda x: x,
    )

    train(train_key, "pendubot", args, p, env, replay_buffer)
