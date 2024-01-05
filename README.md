# Diffusion Co-Policy
Code for [Diffusion Co-Policy for Synergistic Human-Robot Collaborative Tasks](https://arxiv.org/abs/2305.12171).

## Installation
- **Simulation**:      
  First, install the diffusion simulation environment by cloning this repository and running the following commands (pulled from the instructions [here](https://github.com/real-stanford/diffusion_policy?tab=readme-ov-file#%EF%B8%8F-installation)):  
  ```
  sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf`  
  mamba env create -f conda_environment.yaml
  ```
  If you prefer installing via conda, run `conda env create -f conda_environment.yaml`.
  Continue installing the table-carrying simulator in the same environment. After cloning [this](https://github.com/eleyng/table-carrying-ai) repository  and entering the directory, run the following:
  ```
  pip install -r requirements.txt --use-deprecated=legacy-resolver
  pip install -e .
  ```
- **Real robot evaluation**:  
  Further install the ROS environment with instructions [here](https://github.com/armlabstanford/human_robot_transport).

## Dataset
Pull the dataset from Google Drive [here](https://drive.google.com/file/d/1bpsAckORlXIx_C2RnbluSQf4s7hSMpnZ/view?usp=drive_link):
```
curl -L -o data.zip "https://drive.google.com/uc?export=download&id=1bpsAckORlXIx_C2RnbluSQf4s7hSMpnZ"
unzip data.zip; rm data.zip
```

## Training
To train Diffusion Co-policy, launch training with config:
```
(robodiff)[diffusion_copolicy]$ python train.py --config-name=train_diffusion_transformer_lowdim_table_workspace task.dataset_path=data/table_10Hz
```
## Evaluation
TODO: add instructions for sim and real eval

## Acknowledgements
This repository is built on top of [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137), original codebase linked [here](https://github.com/columbia-ai-robotics/diffusion_policy), which is further built on top of other codebases; see repo link for details. We thank the authors for providing us with easily integrable codebases. The codebases for our environments are as follows: the real robot ROS environment is [here](https://github.com/armlabstanford/human_robot_transport), the runtime scripts for both sim and real are [here](https://github.com/eleyng/table-carrying-ai). 
