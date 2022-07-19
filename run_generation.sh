#!/bin/bash

export CARLA_ROOT=carla_server # path to your carla root
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:$(pwd -P)/leaderboard
export PYTHONPATH=$PYTHONPATH:$(pwd -P)/scenario_runner


########### run KING generation ############
# 4 Agents
python generate_scenarios.py \
    --num_agents 4 --save_path ./generation_results/agents_4 --opt_iters 100 --beta1 0.8 --beta2 0.99 --w_adv_col 3.0 --w_adv_rd 20.0

# 2 agents
python generate_scenarios.py \
    --num_agents 2 --save_path ./generation_results/agents_2 --opt_iters 120 --beta1 0.8 --beta2 0.99 --w_adv_col 5.0 --w_adv_rd 23.0

# 1 agent
python generate_scenarios.py \
    --num_agents 1 --save_path ./generation_results/agents_1 --opt_iters 150 --beta1 0.8 --beta2 0.999 --w_adv_col 0.0 --w_adv_rd 20.0


echo "Overall results"
echo "==============="
python3 tools/parse_generation_results.py --results_dir ./generation_results/
echo "4 agents"
echo "==============="
python3 tools/parse_generation_results.py --results_dir ./generation_results/agents_4 --num_agents 4
echo "2 agents"
echo "==============="
python3 tools/parse_generation_results.py --results_dir ./generation_results/agents_2 --num_agents 2
echo "1 agents"
echo "==============="
python3 tools/parse_generation_results.py --results_dir ./generation_results/agents_1 --num_agents 1