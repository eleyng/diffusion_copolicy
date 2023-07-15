import gym
import pygame
import numpy as np
from stable_baselines3.common.env_checker import check_env
import pickle

env = gym.make('cooperative_transport.gym_table:table-v0')
#print("Checking environment!")
#check_env(env, warn=True)

print("Select random actions and step.")
obs = env.reset()
n_steps = env.total_n_step

# load trajectory
#with open(

for n_step in range(n_steps):
    # Random action
    action = np.array([env.curr_traj[n_step][9], env.curr_traj[n_step][11]])
    print('gym action ', action)
    obs, reward, done, info = env.step(action)
    print(info)
    if done:
    	pass
        #obs = env.reset()

print("Check complete. Starting training.", reward)


#env.render()

