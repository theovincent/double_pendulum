import os
from tqdm import tqdm
import numpy as np
import jax
from argparse import Namespace

from dqn.sample_collection.exploration import EpsilonGreedySchedule
from dqn.sample_collection.replay_buffer import ReplayBuffer
from sklearn.tree import DecisionTreeRegressor
from dqn.environments.double_pendulum import PendubotEnv
from dqn.utils.pickle import save_pickled_data
import matplotlib.pyplot as plt


def train(
    key: jax.random.PRNGKey,
    environment_name: str,
    args: Namespace,
    p: dict,
    env: PendubotEnv,
    replay_buffer: ReplayBuffer,
) -> None:
    experiment_path = f"experiments/{environment_name}/figures/{args.experiment_name}/Tree/"

    sample_key, exploration_key = jax.random.split(key)
    js = np.zeros(p["n_epochs"]) * np.nan
    max_j = -float("inf")
    argmax_j = None

    env.collect_random_samples(sample_key, replay_buffer, p["n_initial_samples"], p["horizon"])

    epsilon_schedule = EpsilonGreedySchedule(
        p["starting_eps"],
        p["ending_eps"],
        p["duration_eps"],
        exploration_key,
        0,
    )

    q = DecisionTreeRegressor()

    for idx_epoch in tqdm(range(p["n_epochs"])):
        # learn new tree
        samples = np.hstack((replay_buffer.states, replay_buffer.actions.reshape((-1, 1))))

        next_samples = np.hstack(
            (
                np.repeat(replay_buffer.next_states, env.n_actions, axis=0),
                np.repeat(np.arange(env.n_actions).reshape((1, -1, 1)), replay_buffer.states.shape[0], axis=0).reshape(
                    (-1, 1)
                ),
            )
        )
        if idx_epoch == 0:
            max_q = 0
        else:
            max_q = q.predict(next_samples).reshape((replay_buffer.states.shape[0], env.n_actions)).max(axis=1)
        targets = replay_buffer.rewards + (1 - replay_buffer.absorbings) * p["gamma"] * max_q

        q.fit(samples, targets)

        # collect samples
        sum_rewards = 0

        for _ in range(p["n_training_steps_per_epoch"] // p["horizon"]):
            absorbing = False
            env.reset()
            while not absorbing and env.n_steps < p["horizon"]:
                state = env.state

                if epsilon_schedule.explore():
                    action = jax.random.choice(epsilon_schedule.key, np.arange(env.n_actions)).astype(np.int8)
                else:
                    action = np.argmax(
                        q.predict(
                            np.hstack(
                                (
                                    np.repeat([env.state], env.n_actions, axis=0),
                                    np.arange(env.n_actions).reshape((-1, 1)),
                                )
                            )
                        )
                    )
                next_state, reward, absorbing, _ = env.step(action)

                replay_buffer.add(state, action, reward, next_state, absorbing)
                sum_rewards += reward

        js[idx_epoch] = sum_rewards / (p["n_training_steps_per_epoch"] // p["horizon"])
        np.save(
            f"{experiment_path}J_{args.seed}.npy",
            js,
        )
        if js[idx_epoch] > max_j:
            if argmax_j is not None:
                os.remove(f"{experiment_path}Q_{args.seed}_{argmax_j}_best_online_params")

            argmax_j = idx_epoch
            max_j = js[idx_epoch]
            params = q.get_params()
            save_pickled_data(f"{experiment_path}Q_{args.seed}_{argmax_j}_best_online_params", params)

            # save performance
            absorbing = False
            rewards = []
            q_1 = []
            q_2 = []
            q_prime_1 = []
            q_prime_2 = []
            actions = []
            env.reset()

            while not absorbing and env.n_steps < p["horizon"]:
                action = np.argmax(
                    q.predict(
                        np.hstack(
                            (
                                np.repeat([env.state], env.n_actions, axis=0),
                                np.arange(env.n_actions).reshape((-1, 1)),
                            )
                        )
                    )
                )
                _, reward, absorbing, _ = env.step(action)

                rewards.append(reward)
                q_1.append(env.simulator.x[0])
                q_2.append(env.simulator.x[1])
                q_prime_1.append(env.simulator.x[2])
                q_prime_2.append(env.simulator.x[3])
                actions.append(env.actions[action])

            fig, axes = plt.subplots((5), figsize=(5, 7))

            axes[0].plot(q_1, label="q1")
            axes[0].plot(q_2, label="q2")
            axes[0].legend()
            axes[0].set_title(f"Reward {np.around(sum(rewards), 3)}, n steps {env.n_steps}")
            axes[1].plot(np.cos(q_1), label="cos(q1)")
            axes[1].plot(np.sin(q_1), label="sin(q1)")
            axes[1].legend()
            axes[2].plot(q_prime_1, label="q dot 1")
            axes[2].plot(q_prime_2, label="q dot 2")
            axes[2].legend()
            axes[3].plot(actions, label="u1")
            axes[3].legend()
            axes[4].plot(rewards, label="reward")
            axes[4].legend()
            fig.tight_layout()
            plt.savefig(f"{experiment_path}Q_{args.seed}_{argmax_j}_best_online_params.png")
