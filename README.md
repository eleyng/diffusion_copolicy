# Diffusion Co-Policy
Code for [Diffusion Co-Policy for Synergistic Human-Robot Collaborative Tasks](https://arxiv.org/abs/2305.12171).

## Installation
1. For model training:
  Follow the installation instructions [here](https://github.com/columbia-ai-robotics/diffusion_policy).
2. For simulation evaluation:
  Continue installing in the same environment from step 1, with the instructions [here](https://github.com/eleyng/table-carrying-ai).
3. For real robot evaluation:
  Further install with instructions [here](https://github.com/armlabstanford/human_robot_transport).

## Dataset
TODO: add dataset link

## Training
To train Diffusion Co-policy, launch training with config:
```
(robodiff)[diffusion_copolicy]$ python train.py --config-name=train_diffusion_transformer_lowdim_table_workspace task.dataset_path=data/table_10Hz
```
## Evaluation
TODO: add instructions for sim and real eval

## Acknowledgements
This repository is built on top of [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137), original codebase linked [here](https://github.com/columbia-ai-robotics/diffusion_policy), which is further built on top of other codebases; see repo link for details. We thank the authors for providing us with easily integrable codebases. The codebases for our environments are as follows: the real robot ROS environment is [here](https://github.com/armlabstanford/human_robot_transport), the runtime scripts for both sim and real are [here](https://github.com/eleyng/table-carrying-ai). 
