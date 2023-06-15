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
        # 'bitstring',
        # 'pyyaml',
        # 'argparse',
        # c++ python bindings
        "cython",
        # # optimal control
        # 'drake',
        # 'meshcat',
        # 'IPython',
        # 'filterpy',
        # # 'crocoddyl',
        # # reinforcement learning
        # 'tensorflow>=2.6.0',
        # 'pickle5',
        # 'stable_baselines3'
        # # documentation
        # 'sphinx',
        # 'sphinx-rtd-theme',
        # 'numpydoc',
        # # testing
        # 'pytest',
        # 'lark',
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
            "gym==0.21",
            "stable_baselines3",
        ],
        "doc": ["sphinx", "sphinx-rtd-theme", "numpydoc"],
        "test": ["pytest", "lark"],
        "OC": ["drake", "meshcat"],
        "RL": ["gym==0.21", "stable_baselines3"],
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
