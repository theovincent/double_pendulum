#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

[ -d out/acrobot/$EXPERIMENT_NAME ] || mkdir -p out/acrobot/$EXPERIMENT_NAME

[ -d experiments/acrobot/figures/$EXPERIMENT_NAME ] || mkdir -p experiments/acrobot/figures/$EXPERIMENT_NAME
[ -f experiments/acrobot/figures/$EXPERIMENT_NAME/parameters.json ] || cp experiments/acrobot/parameters.json experiments/acrobot/figures/$EXPERIMENT_NAME/parameters.json
[ -d experiments/acrobot/figures/$EXPERIMENT_NAME/DQN ] || mkdir experiments/acrobot/figures/$EXPERIMENT_NAME/DQN


seed_command="export SLURM_ARRAY_TASK_ID=0"

# DQN
echo "launch train idqn"
train_command="launch_job/acrobot/train_dqn.sh -e $EXPERIMENT_NAME -ns $N_PARALLEL_SEEDS"
tmux send-keys -t train "$seed_command" ENTER "$train_command" ENTER