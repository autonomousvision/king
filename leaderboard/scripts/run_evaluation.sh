export CARLA_ROOT=carla_server # path to your carla root
export KING_ROOT= # path to your KING root
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:scenario_runner
export PYTHONPATH=$PYTHONPATH:$KING_ROOT # path to your king root
export LEADERBOARD_ROOT=leaderboard
export CHALLENGE_TRACK_CODENAME=MAP

# Server Ports
export PORT=2000 # same as the carla server port
export TM_PORT=8000 # port for traffic manager, required when spawning multiple servers/clients

# Agent Paths
export TEAM_AGENT=driving_agents/carla/aim_bev/aim_bev_agent.py # agent
export TEAM_CONFIG=driving_agents/king/aim_bev/model_checkpoints/regular/ # model checkpoint, original
# export TEAM_CONFIG=fine_tuning_results/fine_tuned_ratio_0.4/ # model checkpoint, fine-tuned


###########
# Split 0 #
###########

# Evaluation Setup
export ROUTES=leaderboard/data/routes/t10/t10_combined_split0.xml
export SCENARIOS=leaderboard/data/routes/t10/t10_scenarios.json
export DP_SEED=0 # seed for initializing the locations of background actors
export DEBUG_CHALLENGE=0 # visualization of waypoints and forecasting
export COMPLETION_PERCENT=99 # percentage route distance considered to be complete (NEAT eval: 99)
export SAMPLE_SCENARIOS_RANDOMLY=1 # default (prioritized) or modified (random) sampling (NEAT eval: 1)
export INCREMENT_DP_SEED=0 # whether to change data provider seed each route (NEAT eval: 0)
export SHUFFLE_WEATHER=0 # whether to shuffle the weather each frame for data augmentation (NEAT eval: 0)
export RENDER_PERSPECTIVE=0 # whether to render depth and semantics with CARLA perspective camera
export DISTORT_CAMERAS=0 # whether to add lens distortion to cameras
export TESTTIME_AUGMENT=0 # transfuser test-time augmentation (should be changed to config file)
export RESUME=1
export REPETITIONS=1
export CHECKPOINT_ENDPOINT=./carla_results/t10_split_0.json # results file

mkdir -p "./carla_results/"

echo "Entering leaderboard evaluator..."

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--trafficManagerPort=${TM_PORT} \
--dataProviderSeed=${DP_SEED}

###########
# Split 1 #
###########

# Evaluation Setup
export ROUTES=leaderboard/data/routes/t10/t10_combined_split1.xml
export SCENARIOS=leaderboard/data/routes/t10/t10_scenarios.json
export DP_SEED=0 # seed for initializing the locations of background actors
export DEBUG_CHALLENGE=0 # visualization of waypoints and forecasting
export COMPLETION_PERCENT=99 # percentage route distance considered to be complete (NEAT eval: 99)
export SAMPLE_SCENARIOS_RANDOMLY=1 # default (prioritized) or modified (random) sampling (NEAT eval: 1)
export INCREMENT_DP_SEED=0 # whether to change data provider seed each route (NEAT eval: 0)
export SHUFFLE_WEATHER=0 # whether to shuffle the weather each frame for data augmentation (NEAT eval: 0)
export RENDER_PERSPECTIVE=0 # whether to render depth and semantics with CARLA perspective camera
export DISTORT_CAMERAS=0 # whether to add lens distortion to cameras
export TESTTIME_AUGMENT=0 # transfuser test-time augmentation (should be changed to config file)
export RESUME=1
export REPETITIONS=1
export CHECKPOINT_ENDPOINT=./carla_results/t10_split_1.json # results file

echo "Entering leaderboard evaluator..."

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--trafficManagerPort=${TM_PORT} \
--dataProviderSeed=${DP_SEED}

# parse and aggregate the results
python3 tools/carla_result_parser_t10.py --results ./carla_results/