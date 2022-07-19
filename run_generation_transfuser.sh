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
    --renderer_class CARLA --ego_agent transfuser \
    --ego_agent_ckpt driving_agents/king/transfuser/model_checkpoints/regular/transfuser/ \
    --init_root driving_agents/king/transfuser/king_initializations/initializations_subset \
    --num_agents 4 --save_path ./generation_results_transfuser/agents_4 \
    --opt_iters 100 --beta1 0.8 --beta2 0.99 --w_adv_col 3.0 --w_adv_rd 20.0


echo "Results"
echo "==============="
python3 tools/parse_generation_results.py --results_dir ./generation_results_transfuser/agents_4 --num_agents 4