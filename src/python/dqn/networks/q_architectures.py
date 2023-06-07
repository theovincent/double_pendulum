from typing import Sequence
import flax.linen as nn
import jax

from dqn.networks.base_q import DQN


class DoublePendulumQNet(nn.Module):
    features: Sequence[int]
    n_actions: int

    @nn.compact
    def __call__(self, state):
        x = state
        for n_features in self.features:
            x = nn.relu(nn.Dense(n_features)(x))

        x = nn.Dense(self.n_actions)(x)

        return x


class DoublePendulumDQN(DQN):
    def __init__(
        self,
        state_shape: list,
        n_actions: int,
        gamma: float,
        network_key: jax.random.PRNGKeyArray,
        learning_rate: float,
        n_training_steps_per_online_update: int,
        n_training_steps_per_target_update: int,
    ) -> None:
        super().__init__(
            state_shape,
            n_actions,
            gamma,
            DoublePendulumQNet([200, 200], n_actions),
            network_key,
            learning_rate,
            n_training_steps_per_online_update,
            n_training_steps_per_target_update,
        )