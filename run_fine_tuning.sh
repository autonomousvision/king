#!/bin/bash

export CARLA_ROOT=carla_server # path to your carla root
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:$(pwd -P)/leaderboard
export PYTHONPATH=$PYTHONPATH:$(pwd -P)/scenario_runner


########### fine tune ############
# add path to your dataset here
python driving_agents/king/aim_bev/robust_train.py \
    --train_dataset_path ./aim_bev_regular_data/Shards_Train_RoutesPerScenario_15_10_2021_new/ \
    --mixed_batch_ratio 0.4 --id fine_tuned_ratio_0.4