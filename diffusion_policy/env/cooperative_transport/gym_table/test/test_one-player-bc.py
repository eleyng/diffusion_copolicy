import pdb
import torch
import yaml
import gym
import pygame
import numpy as np
from os.path import join, exists, dirname, abspath
from os import mkdir
from stable_baselines3.common.env_checker import check_env
import sys
sys.path.insert(1, dirname(abspath(__file__)) + '/../models')
from bc import sample_discrete, BCRNN

# Loading parameters
yaml_filepath = join(dirname(__file__), "../config/inference_params.yml")
print('Config path: ', yaml_filepath)

device = torch.device( 'cpu')
#device = torch.device('cpu')
def load_cfg(config_file):
    with open(config_file, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

# Load parameters from config file
cfg = load_cfg(yaml_filepath)
base_dir = cfg["base_dir"]
base_model = cfg["base_model"]
bc_mode = cfg['bc_mode']
single = cfg['single']
jointbc_name = cfg['jointbc_name']
player1bc_name = cfg['player1bc_name']
player2bc_name = cfg['player2bc_name']
n_layers = cfg['n_layers']
n_classes = cfg['n_classes']
mode = cfg["mode"]
train_stats_f = cfg["train_stats_f"]
vis_vae_name = cfg["vis_vae_name"]
horizon = cfg["horizon"]
LSIZE = cfg["LSIZE"]
RSIZE = cfg['RSIZE']
BSIZE = cfg["BSIZE"]
SEQ_LEN = cfg["SEQ_LEN"]
NGAUSS = cfg['NGAUSS']
AGAUSS = cfg['AGAUSS']
ASIZE = cfg['ASIZE']

# Get training data statistics for standardizations
train_stats = dict(np.load(join(dirname(__file__), '..', base_dir, jointbc_name, train_stats_f)))
mean_s = torch.Tensor(train_stats['mean_s']).to(device)
std_s = torch.Tensor(train_stats['std_s']).to(device)

# load model
joint_file = join(dirname(__file__), '..', base_dir, jointbc_name, 'best.tar')
print(joint_file)
assert exists(joint_file), \
    "Joint bc is untrained."

joint_state = torch.load(joint_file  , map_location=torch.device('cpu'))

print("Loading {} at epoch {} "
          "with test loss {}".format(
              'BC', joint_state['epoch'], joint_state['precision']))


# Initialize human model
model = BCRNN(LSIZE, RSIZE, ASIZE, n_layers, device, n_classes)
model.load_state_dict(joint_state['state_dict'], strict=False)
model.eval()

def reset_hidden():
    return torch.zeros(2, 1, 2*RSIZE)

hx = reset_hidden()

# Starting env
env = gym.make('cooperative_transport.gym_table:table-v0')
#print("Checking environment!")
#check_env(env, warn=True)

obs = env.reset()
n_steps = 10001
n_iter = 0
done = False
while not done and n_iter < 2500:
    # Standardize obs
    obs = (torch.tensor(obs[:6]) - mean_s.to(device)) / std_s.to(device)
    print('state', obs)
    with torch.no_grad():
        if n_iter % SEQ_LEN == 0 and n_iter != 0:
            hx = reset_hidden()
        # Random action
        _, hx, probs = model(obs.float().unsqueeze(0).unsqueeze(0), hx)
        action = sample_discrete(probs)
        obs, reward, done, info = env.step(action)
    print(info)
    if done:
    	pass
        #obs = env.reset()
    n_iter += 1


