from typing import Dict
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator
from dqn.networks.base_q import BaseQ
from dqn.sample_collection.exploration import EpsilonGreedySchedule
from dqn.sample_collection.replay_buffer import ReplayBuffer


class PendubotEnv:
    def __init__(self, mpar, dt, n_repeated_actions, integrator, x0, goal, n_actions):
        plant = SymbolicDoublePendulum(model_pars=mpar)
        self.simulator = Simulator(plant)
        self.dt = dt
        self.n_repeated_actions = n_repeated_actions
        self.integrator = integrator
        self.x0 = x0
        self.goal = goal
        self.n_actions = n_actions
        assert self.n_actions % 2 == 1
        self.actions = (
            np.concatenate(
                (-np.logspace(-2, 0, self.n_actions // 2)[::-1], np.array([0]), np.logspace(-2, 0, self.n_actions // 2))
            )
            * 6
        )

    def reset(self):
        self.simulator.set_state(0, self.x0)
        self.simulator.reset_data_recorder()
        self.simulator.record_data(0, np.copy(self.x0), None)
        self.simulator.init_filter(self.x0, self.dt, self.integrator)
        self.n_steps = 0

        x_meas = self.simulator.get_measurement(self.dt)
        x_filt = self.simulator.filter_measurement(x_meas)
        self.state = np.array(
            [
                np.cos(x_filt[0]),
                np.sin(x_filt[0]),
                np.cos(x_filt[1]),
                np.sin(x_filt[1]),
                x_filt[2],
                x_filt[3],
            ]
        )

        return self.state

    def step(self, idx_action):
        reward = 0

        for _ in range(self.n_repeated_actions):
            u = np.array([self.actions[idx_action], 0])
            nu = self.simulator.get_real_applied_u(u)

            self.simulator.step(nu, self.dt, integrator=self.integrator)
            x_meas = self.simulator.get_measurement(self.dt)
            x_filt = self.simulator.filter_measurement(x_meas)
            self.state = np.array(
                [
                    np.cos(x_filt[0]),
                    np.sin(x_filt[0]),
                    np.cos(x_filt[1]),
                    np.sin(x_filt[1]),
                    x_filt[2],
                    x_filt[3],
                ]
            )

            diff_to_goal = self.state[:4] - np.array([-1, 0, 1, 0])
            reward += np.exp(-diff_to_goal @ np.diag([1, 1, 1 / 3, 1 / 3]) @ diff_to_goal.T) * (self.state[0] < -0.2)

        self.n_steps += 1

        return self.state, reward / self.n_repeated_actions, False, {}

    def collect_random_samples(
        self, sample_key: jax.random.PRNGKeyArray, replay_buffer: ReplayBuffer, n_samples: int, horizon: int
    ) -> None:
        self.reset()

        for _ in tqdm(range(n_samples)):
            state = self.state

            sample_key, key = jax.random.split(sample_key)
            idx_action = jax.random.choice(key, jnp.arange(self.n_actions))
            next_state, reward, absorbing, _ = self.step(idx_action)

            replay_buffer.add(state, idx_action, reward, next_state, absorbing)

            if absorbing or self.n_steps >= horizon:
                self.reset()

    def collect_one_sample(
        self,
        q: BaseQ,
        q_params: Dict,
        horizon: int,
        replay_buffer: ReplayBuffer,
        exploration_schedule: EpsilonGreedySchedule,
    ) -> bool:
        state = self.state
        has_reset = False

        if exploration_schedule.explore():
            action = q.random_action(exploration_schedule.key)
        else:
            action = q.best_action(exploration_schedule.key, q_params, self.state)

        next_state, reward, absorbing, _ = self.step(action)

        replay_buffer.add(state, action, reward, next_state, absorbing)

        if absorbing or self.n_steps >= horizon:
            self.reset()
            has_reset = True

        return reward, has_reset or absorbing
