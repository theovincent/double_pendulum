#!/bin/bash


for REPLAY_BUFFER_SIZE in 50000 200000
do
    for DURATION_EPS in 25000 125000
    do 
        for BATCH_SIZE in 64 128 512
        do
            for TARGET_UPDATES in 100 500 1000
            do 
                EXPERIMENT_NAME=r$REPLAY_BUFFER_SIZE\_d$DURATION_EPS\_b$BATCH_SIZE\_t$TARGET_UPDATES

                mkdir examples/reinforcement_learning/DQN/experiments/$EXPERIMENT_NAME/
                PARAMS=$(jq '.replay_buffer_size = $REPLAY_BUFFER_SIZE' --argjson REPLAY_BUFFER_SIZE $REPLAY_BUFFER_SIZE examples/reinforcement_learning/DQN/parameters.json)
                PARAMS=$(jq '.duration_eps = $DURATION_EPS' --argjson DURATION_EPS $DURATION_EPS <<<"$PARAMS")
                PARAMS=$(jq '.batch_size = $BATCH_SIZE' --argjson BATCH_SIZE $BATCH_SIZE <<<"$PARAMS")
                PARAMS=$(jq '.n_training_steps_per_target_update = $TARGET_UPDATES' --argjson TARGET_UPDATES $TARGET_UPDATES <<<"$PARAMS")
                echo $PARAMS > examples/reinforcement_learning/DQN/experiments/$EXPERIMENT_NAME/parameters.json
                            
                sbatch -J $EXPERIMENT_NAME --ntasks=4 --mem-per-cpu=500M --time=05:30:00 --output=examples/reinforcement_learning/DQN/experiments/$EXPERIMENT_NAME/logs.out train_dqn.sh $EXPERIMENT_NAME
            done
        done
    done
done
