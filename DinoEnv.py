import gymnasium as gym
import pygame
from pygame.locals import *
import numpy as np
import DinoGame
import time
import sys


class DinoEnv(gym.Env):
    def __init__(self, renderMode=True):
        super().__init__()

        self.renderMode = renderMode
        if self.renderMode:
            # Initialize the game window
            pygame.init()
            self.screen = pygame.display.set_mode((DinoGame.SCREEN_WIDTH, DinoGame.SCREEN_HEIGHT))
            pygame.display.set_caption('Dino Game')
            self.image_width = DinoGame.BG.get_width()
            self.font = pygame.font.Font('freesansbold.ttf', 20)


        self.action_space = gym.spaces.Discrete(2)
        # self.observation_space = gym.spaces.Box(low=0, high=DinoGame.SCREEN_WIDTH, shape=(2,), dtype=np.int16)
        self.observation_space = gym.spaces.Dict({
            'should_perform': gym.spaces.Discrete(2),
            'is_jumping': gym.spaces.Discrete(2),
        })

        self.clock = pygame.time.Clock()
        self.player = DinoGame.Dinosaur()
        self.cloud = DinoGame.Cloud()

        self.avoided_obstacle = False

        
    def reset(self, seed=None, options=None):
        # Reset the game environment
        self.game_speed = 20
        self.x_pos_bg = 0
        self.y_pos_bg = 380
        self.points = 0
        self.obstacles = []
        self.player.reset()

        self.obstacles.append(DinoGame.generate_obstacle())
        
        return self._get_observation(), {}
    
    def step(self, action):
        prev_obs = self._get_observation()
        # Perform the given action
        userInput = np.zeros(np.max([pygame.K_UP, pygame.K_DOWN]) + 1, dtype=bool)
        if action == 1:
            userInput[pygame.K_UP] = True
        elif action == 0:
            userInput[pygame.K_DOWN] = True
        
        # Update the game environment
        self.player.update(userInput, self.game_speed)
        for obstacle in self.obstacles:
            obstacle.update(self.game_speed)
            if obstacle.rect.x < -obstacle.rect.width:
                self.obstacles.pop()

        if len(self.obstacles) <= 0:
            self.obstacles.append(DinoGame.generate_obstacle())
            self.avoided_obstacle = False

        self.cloud.update(self.game_speed)
        self.x_pos_bg -= self.game_speed

        self.points += 1
        if self.points % 100 == 0:
            self.game_speed += 1

        # Calculate the reward, done flag, and info
        reward = 1 if prev_obs['should_perform'] == 1 else 0
        done = False
        info = {}


        if self.player.dino_rect.colliderect(self.obstacles[0].rect):
            reward = -5
            done = True
        
        return self._get_observation(), reward, done, False, info
    
    def render(self):
        if not self.renderMode:
            return
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
        
        pygame.event.pump()

        # Render the game environment
        self.screen.fill((255, 255, 255))

        self.player.draw(self.screen)
        for obstacle in self.obstacles:
            obstacle.draw(self.screen)
        self.cloud.draw(self.screen)

        self.screen.blit(DinoGame.BG, (self.x_pos_bg, self.y_pos_bg))
        self.screen.blit(DinoGame.BG, (self.image_width + self.x_pos_bg, self.y_pos_bg))
        if self.x_pos_bg <= -self.image_width:
            self.screen.blit(DinoGame.BG, (self.image_width + self.x_pos_bg, self.y_pos_bg))
            self.x_pos_bg = 0

        text = self.font.render("Points: " + str(self.points), True, (0, 0, 0))
        textRect = text.get_rect()
        textRect.center = (1000, 40)
        self.screen.blit(text, textRect)

        pygame.display.update()
        self.clock.tick(30)


    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            sys.exit()
        
    
    def _get_observation(self):
        # return np.array([self.obstacles[0].rect.x, self.obstacles[0].rect.y], dtype=np.int16)
        o_x = self.obstacles[0].rect.x
        return {
                'should_perform': 1 if o_x <= self.player.dino_rect.x + self.player.dino_rect.width*2 + 30 + self.game_speed**3/1000 and self.obstacles[0].rect.y > 250 else 0,
                'is_jumping': 1 if self.player.dino_jump else 0,
            }


env = DinoEnv()
while True:
    observation, _ = env.reset()
    done = False
    while not done:
        # action = env.action_space.sample()  # Randomly select an action

        # action = 0
        # distanceToObs = observation['obs_position_x'] - (env.player.dino_rect.x + env.player.dino_rect.width + observation['game_speed']**2/10)
        # print(1 if observation['obs_position_x'] <= env.player.dino_rect.x + env.player.dino_rect.width*2 + 20 + observation['game_speed']**2/10 and observation['obs_position_y'] > 250 else 0)
        # shouldPerformAction = distanceToObs <= 50 and distanceToObs >= 0
        # if shouldPerformAction:
        #     if observation['obs_position_y'] > 250:
        #         action = 1

        action = 1 if observation['should_perform'] else 0
        observation, reward, done, _, info = env.step(action)
        print(observation, reward, done, info)
        env.render()
    time.sleep(1)
