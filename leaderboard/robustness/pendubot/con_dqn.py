import json

from double_pendulum.controller.DQN.dqn import DQNController
from double_pendulum.utils.plotting import plot_timeseries

from sim_parameters import mpar, goal, x0, dt, integrator, design, robot


experiment_name = "first"
best_params = 86
name = "dqn_" + experiment_name
leaderboard_config = {
    "csv_path": name + "/sim_swingup.csv",
    "name": name,
    "simple_name": "DQN",
    "short_description": "DQN.",
    "readme_path": f"readmes/{name}.md",
    "username": "theovincent",
}

# # model parameters
torque_limit = [6.0, 0.0]

mpar.set_motor_inertia(0.0)
mpar.set_damping([0.0, 0.0])
mpar.set_cfric([0.0, 0.0])
mpar.set_torque_limit(torque_limit)

# controller parameters
N = 400
N_init = 400
max_iter = 5
max_iter_init = 100
regu_init = 1.0
max_regu = 10000.0
min_regu = 0.0001
break_cost_redu = 1e-6
trajectory_stabilization = False

f_sCu = [0.1, 0.1]
f_sCp = [10.0, 0.1]
f_sCv = [0.05, 0.2]
f_sCen = 0.0
f_fCp = [50.0, 10.0]
f_fCv = [1.0, 1.0]
f_fCen = 1.0

# construct simulation objects
p = json.load(
    open(f"../../../src/python/experiments/pendubot/figures/{experiment_name}/parameters.json")
)  # p for parameters
controller = DQNController(
    f"../../../src/python/experiments/pendubot/figures/{experiment_name}/DQN/Q_10_{best_params}_best_online_params",
    p["n_actions"],
    p["n_repeated_actions"],
    p["layers_dimensions"],
    torque_limit,
)
controller.set_start(x0)
controller.set_goal(goal)


# T, X, U = controller.get_init_trajectory()
# plot_timeseries(T, X, U)

controller.init()
