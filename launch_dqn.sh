#!/bin/bash

sbatch -J $@ --ntasks=4 --mem-per-cpu=500M --time=01:00:00 --output=examples/reinforcement_learning/DQN/experiments/$@/logs.out train_dqn.sh $@
