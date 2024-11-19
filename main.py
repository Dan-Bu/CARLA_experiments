import sys
import os
import carla
from carla import command
import random
import numpy as np
from numpy import random
import cv2
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from PIL import Image
from tqdm import tqdm
from threading import Thread

from torchvision import transforms
from torch.utils.data import DataLoader
import datetime
import re

import settings
import generateTraffic
import generateTrainingImages
import dataloader.rgbcameradataloader as customDataset
import networks.camera.base_model as base_model
import cameraImageSegmentation
import reinforcement_learning.DQNAgent as DQNAgent
import reinforcement_learning.RLDrivingEnvironment as RLDrivingEnvironment


#sys.path.append('C:\\Users\\Daniel\\Documents\\PrivateProjects\\CARLA\\CARLA_0.9.15\\WindowsNoEditor\\PythonAPI\\carla') # tweak to where you put carla
sys.path.append('C:/Users/Daniel/Documents/PrivateProjects/CARLA/CARLA_0.9.15/WindowsNoEditor/PythonAPI/carla') # tweak to where you put carla
from agents.navigation.global_route_planner import GlobalRoutePlanner







def train_camera_driving():
    # We need an agent and an environment for this
    agent = DQNAgent.Agent()
    env = RLDrivingEnvironment.vehicleEnv()

    # Fix seeding for experiment reproducability
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    FPS = 60

    # For stat tracking
    ep_rewards = [-200]

    # Create models folder for saving
    if not os.path.isdir('RLmodels'):
        os.makedirs('RLmodels')

    # Start training thread and wait for training to be initialized. Every 1000 steps this thread will then train our network.
    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()
    while not agent.training_initialized:
        time.sleep(0.01)

    # The agent has to make the predictions, the environment will tell us the states we are in.
    # This will happen in a loop.
    for episode in tqdm(range(1, DQNAgent.NUM_EPISODES + 1), ascii=True, unit='episodes'):
        #Set reward for this episode to 0, step to 1, reset the environment and set done and time.
        episode_reward = 0
        step = 1

        current_state = env.reset()

        done = False
        episode_start = time.time()

        cv2.namedWindow('RGB Camera', cv2.WINDOW_AUTOSIZE)
    
        # Play for given number of seconds only
        while True:

            # This part stays mostly the same, the change is to query a model for Q values
            if np.random.random() > DQNAgent.EPSILON:
                # Get action from Q table
                action = np.argmax(agent.get_qs(current_state))
            else:
                # Get random action
                action = np.random.randint(0, 4)
                # This takes no time, so we add a delay matching 60 FPS (prediction above takes longer)
                time.sleep(1/FPS)

            new_state, reward, done, _ = env.step(action)

            # Transform new continous state to new discrete state and count reward
            episode_reward += reward

            # Every step we update replay memory
            agent.update_replay_memory((current_state, action, reward, new_state, done))

            current_state = new_state
            step += 1

            #show camera feed
            cv2.imshow('RGB Camera', env.front_camera_feed)
            if cv2.waitKey(10)==ord('q'):
                return
            cv2.imshow('SemSeg Camera', env.semseg_front_camera_feed)
            if cv2.waitKey(10)==ord('q'):
                return

            if done:
                break
        
        # End of episode - destroy agents
        for actor in env.actor_list:
            actor.destroy()


        ep_rewards.append(episode_reward)

        if not episode % DQNAgent.AGGREGATE_STATS_EVERY or episode == 1:
                average_reward = sum(ep_rewards[-DQNAgent.AGGREGATE_STATS_EVERY:])/len(ep_rewards[-DQNAgent.AGGREGATE_STATS_EVERY:])
                min_reward = min(ep_rewards[-DQNAgent.AGGREGATE_STATS_EVERY:])
                max_reward = max(ep_rewards[-DQNAgent.AGGREGATE_STATS_EVERY:])
                agent.tensorboard.add_scalars("Step_Logging", {"reward_avg":average_reward, "reward_min":min_reward, "reward_max":max_reward, "epsilon":agent.epsilon}, episode)

                # Save model, but only when min reward is greater or equal a set value
                # Actually, saving them every time will take up far too much space on my hardware...
                #if min_reward >= RLDrivingEnvironment.MIN_REWARD:
                #    torch.save(agent.model.state_dict(), f'./RLmodels/{DQNAgent.MODEL_NAME}__Trained__{DQNAgent.NUM_EPISODES}eps')

        # Decay Epsilon to slowly use the network more while never fully losing random exploration
        if agent.epsilon > DQNAgent.MIN_EPSILON:
                agent.epsilon *= DQNAgent.EPSILON_DECAY
                agent.epsilon = max(DQNAgent.MIN_EPSILON, agent.epsilon)



    #Terminate training thread and save model.
    agent.terminate = True
    trainer_thread.join()
    torch.save(agent.model.state_dict(), f'RLmodels/{DQNAgent.MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}')

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium') # To utilize the 4070 TI Supers tensor cores
    print(f"CUDA version: {torch.version.cuda}")
    if settings.mode == "test_single_image":
        cameraImageSegmentation.test_on_single_image()
        exit()

    if settings.mode == "GenerateData":
        generateTrainingImages.generate_training_data()
        exit()

    if settings.mode == "CameraSegmentationTestrun":
        cameraImageSegmentation.camera_segmentation_testrun()
        exit()
        
    if settings.mode == "TrainSemUNet":
        cameraImageSegmentation.train_UNet()
        exit()

    if settings.mode == "TrainCameraDriving":
        train_camera_driving()
        exit()

    if settings.mode == "CameraDriving":

        exit()
