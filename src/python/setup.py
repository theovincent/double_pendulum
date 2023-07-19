from setuptools import setup, find_packages

setup(
    name="DoublePendulum",
    author="Underactuated Lab DFKI Robotics Innovation Center Bremen",
    version="1.0.0",
    url="https://github.com/dfki-ric-underactuated-lab",
    packages=find_packages(),
    install_requires=[
        # general
        "numpy",
        "matplotlib",
        "pandas",
        "scipy",
        "sympy",
        "scikit-learn",
        "cma",
        "lxml",
        "mini-cheetah-motor-driver-socketcan",
        "moteus",
        "inputs",
        "tabulate",
        "filterpy",
        "dill",
        "argparse",
        "opencv-python",
        # c++ python bindings
        "cython==0.29.36",
    ],
    extras_require={
        "all": [
            "sphinx",
            "sphinx-rtd-theme",
            "numpydoc",
            "pytest",
            "lark",
            "drake",
            "meshcat",
            "gymnasium",
            "stable_baselines3",
            "shimmy",
        ],
        "doc": ["sphinx", "sphinx-rtd-theme", "numpydoc"],
        "test": ["pytest", "lark"],
        "OC": ["drake", "meshcat"],
        "RL": ["gymnasium", "stable_baselines3", "shimmy"],
    },
    classifiers=[
        "Development Status :: 5 - Stable",
        "Environment :: Console",
        "Intended Audience :: Academic Usage",
        "Programming Language :: Python",
    ],
    entry_points={
        "console_scripts": [
            "parallel_launcher = launch_job.parallel_launcher:run_cli",
            "acrobot_dqn = experiments.acrobot.DQN:run_cli",
            "pendubot_dqn = experiments.pendubot.DQN:run_cli",
            "pendubot_tree = experiments.pendubot.Tree:run_cli",
        ],
    },
)
