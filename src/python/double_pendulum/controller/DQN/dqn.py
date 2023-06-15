import os
import yaml
import numpy as np

from double_pendulum.controller.abstract_controller import AbstractController
from dqn.utils.pickle import load_pickled_data
from dqn.networks.q_architectures import DoublePendulumDQN
import jax


class DQNController(AbstractController):
    """DQN Controller

    DQN controller

    Parameters
    ----------
    model_pars : model_parameters object, optional
        object of the model_parameters class, default=None
        Can be used to set all model parameters above
        If provided, the model_pars parameters overwrite
        the other provided parameters
        (Default value=None)
    """

    def __init__(self, model_pars, n_actions, n_repeated_actions, layers_dimensions, torque_limit):
        super().__init__()

        self.q = DoublePendulumDQN((6,), n_actions, None, layers_dimensions, jax.random.PRNGKey(0), 0, 0, 0)
        self.q.params = load_pickled_data(model_pars)

        self.counter = 0
        self.previous_action = None
        self.n_repeated_actions = n_repeated_actions

        self.torque_limit = torque_limit
        if self.torque_limit[0] > 0.0:
            self.active_act = 0
        elif self.torque_limit[1] > 0.0:
            self.active_act = 1

        self.actions = (
            np.concatenate(
                (
                    -np.logspace(-2, 0, self.q.n_actions // 2)[::-1],
                    np.array([0]),
                    np.logspace(-2, 0, self.q.n_actions // 2),
                )
            )
            * 6
        )

        # set default parameters
        self.set_start()
        self.set_goal()
        self.set_parameters()

    def set_start(self, x=[0.0, 0.0, 0.0, 0.0]):
        """set_start
        Set start state for the controller.

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
            (Default value=[0., 0., 0., 0.])
        """
        self.start = np.asarray(x)

    def set_goal(self, x=[np.pi, 0.0, 0.0, 0.0]):
        """set_goal.
        Set goal for the controller.

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
            (Default value=[np.pi, 0., 0., 0.])
        """
        self.goal = np.asarray(x)

    def init_(self):
        """
        Initalize the controller.
        """
        pass

    def get_control_output_(self, x, t=None):
        """
        The function to compute the control input for the double pendulum's
        actuator(s).

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
        t : float, optional
            time, unit=[s]
            (Default value=None)

        Returns
        -------
        array_like
            shape=(2,), dtype=float
            actuation input/motor torque,
            order=[u1, u2],
            units=[Nm]
        """
        x_extended = np.array([np.cos(x[0]), np.sin(x[0]), np.cos(x[1]), np.sin(x[1]), x[2], x[3]])

        if self.counter % self.n_repeated_actions == 0:
            self.previous_action = np.argmax(self.q(self.q.params, x_extended))
        u_act = self.actions[self.previous_action]

        # u = [self.u1_traj[0], self.u2_traj[0]]
        u = [0.0, 0.0]
        u[self.active_act] = u_act
        # u = [0.0, u_act]

        u[0] = np.clip(u[0], -self.torque_limit[0], self.torque_limit[0])
        u[1] = np.clip(u[1], -self.torque_limit[1], self.torque_limit[1])

        self.counter += 1
        return u
