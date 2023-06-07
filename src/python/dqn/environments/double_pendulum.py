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


class AcrobotEnv:
    def __init__(self, mpar, dt, integrator, x0, goal, n_actions):
        plant = SymbolicDoublePendulum(model_pars=mpar)
        self.simulator = Simulator(plant)
        self.dt = dt
        self.integrator = integrator
        self.x0 = x0
        self.goal = goal
        
        sCu = [1.0, 1.0]
        sCp = [1.0, 1.0]
        sCv = [0.01, 0.01]
        self.Q = np.array(
            [
                [sCp[0], 0.0, 0.0, 0.0],
                [0.0, sCp[1], 0.0, 0.0],
                [0.0, 0.0, sCv[0], 0.0],
                [0.0, 0.0, 0.0, sCv[1]],
            ]
        )
        self.R = np.array([[sCu[0], 0.0], [0.0, sCu[1]]])
        self.n_actions = n_actions
        # change 1 to 0 for pendulum
        self.actions = np.linspace(-plant.torque_limit[1], plant.torque_limit[1], self.n_actions)

    def reset(self):
        self.simulator.set_state(0, self.x0)
        self.simulator.reset_data_recorder()
        self.simulator.record_data(0, np.copy(self.x0), None)
        self.simulator.init_filter(self.x0, self.dt, self.integrator)
        self.n_steps = 0

        x_meas = self.simulator.get_measurement(self.dt)
        self.state = self.simulator.filter_measurement(x_meas)

        return self.state

    def step(self, idx_action):
        # exchange indexes for pendulum
        u = np.array([0, self.actions[idx_action]])
        nu = self.simulator.get_real_applied_u(u)

        self.simulator.step(nu, self.dt, integrator=self.integrator)
        self.n_steps += 1
        x_meas = self.simulator.get_measurement(self.dt)
        self.state = self.simulator.filter_measurement(x_meas)

        x_diff = self.state - self.goal

        cost = x_diff @ self.Q @ x_diff.T + nu @ self.R @ nu.T

        return self.state, -cost, False, {}
    
    def simulate(self, N, agent):
        self.reset()
        sum_reward = 0

        while self.n_steps < N:
            idx_action = agent.act(self.x_filt)
            _, reward, _, _ = self.step(idx_action)
            sum_reward += reward
        
        return sum_reward
    
    def collect_random_samples(self, sample_key: jax.random.PRNGKeyArray, replay_buffer: ReplayBuffer, n_samples: int, horizon: int
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

        if self.n_steps >= horizon:
            self.reset()
            has_reset = True

        return reward, has_reset

