#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

[ -d out/pendubot/$EXPERIMENT_NAME ] || mkdir -p out/pendubot/$EXPERIMENT_NAME

[ -d experiments/pendubot/figures/$EXPERIMENT_NAME ] || mkdir -p experiments/pendubot/figures/$EXPERIMENT_NAME
[ -f experiments/pendubot/figures/$EXPERIMENT_NAME/parameters.json ] || cp experiments/pendubot/parameters.json experiments/pendubot/figures/$EXPERIMENT_NAME/parameters.json
[ -d experiments/pendubot/figures/$EXPERIMENT_NAME/DQN ] || mkdir experiments/pendubot/figures/$EXPERIMENT_NAME/DQN


seed_command="export SLURM_ARRAY_TASK_ID=0"

# DQN
echo "launch train idqn"
train_command="launch_job/pendubot/train_dqn.sh -e $EXPERIMENT_NAME -ns $N_PARALLEL_SEEDS"
tmux send-keys -t train "$seed_command" ENTER "$train_command" ENTER