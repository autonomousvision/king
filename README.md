# KING: Generating Safety-Critical Driving Scenarios for Robust Imitation via Kinematics Gradients

## [Project Page](https://lasnik.github.io/king/) | [Paper](https://arxiv.org/pdf/2204.13683.pdf) | [Supplementary](https://lasnik.github.io/king/data/docs/KING_supplementary.pdf)

<div style="text-align: center">
  <img style="border:5px solid #263b50;" src="./assets/animated_teaser_h264.gif"/>
</div>
<!-- <img src="./assets/animated_teaser_h264.gif"> -->

This repository contains the code for the ECCV 2022 paper [KING: Generating Safety-Critical Driving Scenarios for Robust Imitation via Kinematics Gradients](https://arxiv.org/pdf/2204.13683.pdf). If you find this repository useful, please cite
```bibtex
@inproceedings{Hanselmann2022ECCV,
  author = {Hanselmann, Niklas and Renz, Katrin and Chitta, Kashyap and Bhattacharyya, Apratim and Geiger, Andreas},
  title = {KING: Generating Safety-Critical Driving Scenarios for Robust Imitation via Kinematics Gradients},
  booktitle = {European Conference on Computer Vision(ECCV)},
  year = {2022}
}
```

## Contents
1. [Setup](#setup)
3. [Scenario Generation](#king-scenario-generation)
4. [Fine-tuning](#fine-tuning)
5. [Town10 Intersections](#town10-intersections)
7. [Acknowledgements](#acknowledgements)

## Setup
Install anaconda
```Shell
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
bash Anaconda3-2022.05-Linux-x86_64.sh
source ~/.profile
```

Clone the repo and build the environment

```Shell
git clone https://github.com/autonomousvision/king
cd king
conda env create -f environment.yml
conda activate king
```

Download and setup CARLA 0.9.10.1
```Shell
chmod +x setup_carla.sh
./setup_carla.sh
```

## Running the code
We provide bash scripts for the experiments for convenience. Please make sure the "CARLA_ROOT" ("./carla_server" by default) and "KING_ROOT" (if present) environment variables are set correctly in all of those scripts.

### KING Scenario Generation
For all of the generation scripts, first spin up a carla server in a separate shell:
```Shell
carla_server/CarlaUE4.sh --world-port=2000 -opengl
```
Then run the following script for AIM-BEV generation:
```Shell
bash run_generation.sh
```
This script runs generation for all traffic density and automatically evaluates the results.
For AIM-BEV generation using both gradient paths, run:
```Shell
bash run_generation_both_paths.sh
```
Finally, to generate scenarios for [TransFuser](https://github.com/autonomousvision/transfuser), first download the model weights:
```Shell
mkdir driving_agents/king/transfuser/model_checkpoints/regular
cd driving_agents/king/transfuser/model_checkpoints/regular
wget https://s3.eu-central-1.amazonaws.com/avg-projects/transfuser/models.zip
unzip models.zip
rm -rf models.zip late_fusion geometric_fusion cilrs aim
```
Then change back to root directory of the repository and run:
```Shell
bash run_generation_transfuser.sh
```

### Fine-tuning
To fine-tune the original agent on KING scenarios, first download the regular data for AIM-BEV:
```
chmod +x download_regular_data.sh
./download_regular_data.sh
```
Then run the fine-tuning script (and adjust it for the correct dataset path, if necessary):
```
bash run_fine_tuning.sh
```
This script also automatically runs evaluation on KING scenarios.

### Town10 Intersections 
To evaluate the original checkpoint for AIM-BEV on the Town10 intersections benchmark, spin up a carla server and run
```Shell
leaderboard/scripts/run_evaluation.sh
```
To evaluate a fine-tuned model, run the fine-tuning script above and toggle the commented "TEAM_CONFIG" variable in the evaluation script to change the model weights.

## Acknowledgements
This implementation is based on code from several repositories. We sincerely thank the authors for their awesome work.
- [CARLA Leaderboard](https://github.com/carla-simulator/leaderboard)
- [Scenario Runner](https://github.com/carla-simulator/scenario_runner)
- [Learning by Cheating](https://github.com/dotchen/LearningByCheating)
- [World on Rails](https://github.com/dotchen/WorldOnRails)

Also, check out the code for other recent work on CARLA from our group:
- [Renz et al., PlanT: Explainable Planning Transformers via Object-Level Representations (CoRL 2022)](https://github.com/autonomousvision/plant)
- [Chitta et al., TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving (PAMI 2022)](https://github.com/autonomousvision/transfuser)
- [Chitta et al., NEAT: Neural Attention Fields for End-to-End Autonomous Driving (ICCV 2021)](https://github.com/autonomousvision/neat)
