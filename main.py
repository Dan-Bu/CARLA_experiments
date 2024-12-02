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

    # Set up spectator camera
    spectator = env.world.get_spectator()
    transform = carla.Transform(carla.Location(x=100, y=210, z=180), carla.Rotation(pitch = -90.0, yaw = 90))
    spectator.set_transform(transform)
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
            if np.random.random() > agent.epsilon:
                # Get action from Q table
                result = (agent.get_qs(current_state))
                result_max = result.max(1)
                action = int(result_max[1][0])
            else:
                # Get random action
                action = np.random.randint(0, DQNAgent.MODEL_OUTPUT_SIZE - 1)
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

            # Draw current route
            env.draw_route(env.current_route)

            if done:
                break

        episode_end = time.time()
        agent.tensorboard.add_scalar("Episode_Logging/episode_time", episode_end - episode_start, episode)

        # End of episode - destroy agents
        for actor in env.actor_list:
            actor.destroy()


        ep_rewards.append(episode_reward)

        if not episode % DQNAgent.AGGREGATE_STATS_EVERY or episode == 1:
                average_reward = sum(ep_rewards[-DQNAgent.AGGREGATE_STATS_EVERY:])/len(ep_rewards[-DQNAgent.AGGREGATE_STATS_EVERY:])
                min_reward = min(ep_rewards[-DQNAgent.AGGREGATE_STATS_EVERY:])
                max_reward = max(ep_rewards[-DQNAgent.AGGREGATE_STATS_EVERY:])
                agent.tensorboard.add_scalars("Step_Logging", {"reward_avg":average_reward, "reward_min":min_reward, "reward_max":max_reward, "epsilon":agent.epsilon}, episode)

                # Save model periodically
                if episode % 1000 == 0:
                    torch.save(agent.model.state_dict(), f'./RLmodels/{DQNAgent.MODEL_NAME}__output__{DQNAgent.MODEL_OUTPUT_SIZE}__eps__{episode}__{datetime.datetime.now().hour}:{datetime.datetime.now().minute}')

        # Decay Epsilon to slowly use the network more while never fully losing random exploration
        if agent.epsilon > DQNAgent.MIN_EPSILON:
                agent.epsilon *= DQNAgent.EPSILON_DECAY
                agent.epsilon = max(DQNAgent.MIN_EPSILON, agent.epsilon)



    #Terminate training thread and save model.
    agent.terminate = True
    trainer_thread.join()
    torch.save(agent.model.state_dict(), f'RLmodels/{DQNAgent.MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__output{DQNAgent.MODEL_OUTPUT_SIZE}__{datetime.datetime.now()}__TotalEps__{episode}')

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
