# Diffusion Co-Policy
Code for [Diffusion Co-Policy for Synergistic Human-Robot Collaborative Tasks](https://arxiv.org/abs/2305.12171).

## Installation
- **Training**:      
  First, install the diffusion simulation environment by cloning this repository and running the following commands (pulled from the instructions [here](https://github.com/real-stanford/diffusion_policy?tab=readme-ov-file#%EF%B8%8F-installation)):  
  ```
  sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf`  
  mamba env create -f conda_environment.yaml
  conda activate robodiff
  ```
- **Evaluation**
  The [table-carrying gym environment](https://github.com/eleyng/table-carrying-ai) is required for both simulation and real robot evaluation. **Clone the `coop-ddpm` branch** of the repository in an external directory. The `conda_environment.yml` file already includes required dependencies.
  
- **Real robot evaluation**:  
  Further install the ROS environment with instructions [here](https://github.com/armlabstanford/human_robot_transport).

## Dataset
Pull the dataset from Google Drive [here (Note: Updated Aug 2024)](https://drive.google.com/file/d/1x47CwVTDAy9sFHVCcgah82L6qKY8zObP/view?usp=sharing):
```
curl -L -o data.zip "https://drive.google.com/uc?export=download&id=1bpsAckORlXIx_C2RnbluSQf4s7hSMpnZ"
unzip data.zip; rm data.zip
```

## Training
To train Diffusion Co-policy (used for both simulation and real environment), launch training with config:
```
python train.py --config-dir=. --config-name=train_diffusion_transformer_lowdim_table_workspace training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
```
The trained model checkpoint file will appear in `data/outputs/table_lowdim` under the training directory created at the particular date-time the run was started. 

### Trained models
Trained models for diffusion co-policy (10Hz), VRNN planner, Co-GAIL, and BC-LSTM-GMM can be downloaded from [here](https://drive.google.com/file/d/129KNvW2HmTgSpb3WBAWFuqCvwQ-cJLsl/view?usp=drive_link) into the table-carrying-ai directory (as `table-carrying-ai/trained_models`):
```
curl -L -o trained_models.zip "https://drive.google.com/uc?export=download&id=129KNvW2HmTgSpb3WBAWFuqCvwQ-cJLsl"
unzip trained_models.zip; rm trained_models.zip
```

## Evaluation
Evaluation scripts were tested on a PC with NVIDIA CUDA-enabled 3090Ti GPU, Ubuntu 22.04, and ROS Noetic.
### HW Requirements for real-human evaluation (sim. env.)  
- USB game controller
### HW Requirements for real-human evaluation (real. env.)  
- 2x [Interbotix Locobot wx250](https://www.trossenrobotics.com/locobot-wx250.aspx)  
- Motion capture system  
- Pin rod and mechanical fasteners (to assemble pin-rod connection between locobots)

For simulation, the trained diffusion co-policy must be evaluated in the table-carrying-ai directory. In the `table-carrying-ai` directory, move the trained checkpoints to the `trained_models` directory and run the following:    
```
python scripts/test_model.py --run-mode [coplanning | replay_traj | hil] --robot- planner --human-mode [planner | data | real] --human-control joystick --human-act-as-cond --subject-id 0 --render-mode headless --planner-type diffusion_policy --map-config cooperative_transport/gym_table/config/maps/varied_maps_${test_set}.yml
```  
where {test_set} can be "test_holdout" or "unseen_map".  
To test other planner types and policies, follow the commands outlined in the `table_carrying_ai/scripts/experiments_[name].sh` scripts where the experiment `[name]` can be one of [replay | hil_traj | coplanning]. "hil_traj" refers to playing with a real human-in-the-loop, "replay" refers to playing with a robot replaying a pre-recorded trajectory, and coplanning refers to robot play.

### Real robot
To eval on real robot, you can run the script:  
``
python scripts/test_model_real.py --run-mode hil --robot-mode planner --human-mode real --human-control joystick --subject-id 0  --human-act-as-cond --render-mode gui --planner-type ${type}  --map-config datasets/real_test/real_test.yml --test-idx ${idx}
``
after setting up the real robot ROS env and launching necessary nodes (for mocap, robots, etc.) as outlined in the instructions [here](https://github.com/armlabstanford/human_robot_transport).  
If you wish to run CoDP-H, ensure the flag `--human-act-as-cond` is called; else, ignore it (for Co-DP). The ROS node running the policy code is in `libs.hil_real_robot_diffusion.play_hil_planner`, and task-specific parameters such as floor dimensions for scaling between sim and real, robot names, etc. are defined in the config `libs.real_robot_utils`.

See the commands outlined in the `table_carrying_ai/scripts/experiments_real.sh` for references in how to call real robot evaluation for the other policies.

## Acknowledgements
This repository is built on top of [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137), original codebase linked [here](https://github.com/columbia-ai-robotics/diffusion_policy), which is further built on top of other codebases; see repo link for details. We thank the authors for providing us with easily integrable codebases. The codebases for our environments are as follows: the real robot ROS environment is [here](https://github.com/armlabstanford/human_robot_transport), the runtime scripts for both sim and real are [here](https://github.com/eleyng/table-carrying-ai). 
