from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

import os 
# Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback
# Check Environment    
from stable_baselines3.common import env_checker

import time
import DinoEnv

env = DinoEnv.DinoEnv(renderMode=True)


model = DQN('MultiInputPolicy', env)

model.load('train2_DQN/best_model_1000000') 

for episode in range(5): 
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done: 
        action, _ = model.predict(obs)
        obs, reward, done, _,  info = env.step(action)
        print(obs, reward, action)
        env.render()
        total_reward += reward
    time.sleep(1)
    print('Total Reward for episode {} is {}'.format(episode, total_reward))