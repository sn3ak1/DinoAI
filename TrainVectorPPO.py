from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env


import os 
# Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback
# Check Environment    
from stable_baselines3.common import env_checker

from DinoEnv import DinoEnv


class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
        return True
    

CHECKPOINT_DIR = './train2_PPO/'
LOG_DIR = './logs/'


if __name__ == '__main__':
    env_lambda = lambda: DinoEnv(renderMode=False)
    num_cpu = 4
    env = SubprocVecEnv([env_lambda for _ in range(num_cpu)])

    model = PPO('MultiInputPolicy',env,verbose=1)

    callback = TrainAndLoggingCallback(check_freq=100000, save_path=CHECKPOINT_DIR)
    model.learn(total_timesteps=1000000, callback=callback)
