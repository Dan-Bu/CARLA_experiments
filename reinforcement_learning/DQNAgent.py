import glob
import os
import sys
import carla
from carla import command
import random
import time
import numpy as np
import settings
import cv2
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import tensorboard
from torch.utils.tensorboard import SummaryWriter

import reinforcement_learning.decisionModel as decisionModel

REPLAY_MEMORY_SIZE = 5000
MIN_REPLAY_MEMORY_SIZE = 100
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
DISCOUNT_FACTOR = 0.99
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 8
UPDATE_TARGET_EVERY_N_STEPS = 5
MODEL_NAME = "UNetDecision"
NUM_EPISODES = 2000
LEARNING_RATE = 0.001
EPSILON = 1 # This is the chance to take a random action (Random exploration) vs doing a prediction with our network.
MIN_EPSILON = 0.01 # Always have a base chance of 1% to take a random action to allow for exploration always
EPSILON_DECAY = 0.95 # Decay for the epsilon greedy policy. Slowly decays epsilon to give the network more responsibility instead of choosing random actions

AGGREGATE_STATS_EVERY = 10

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)


class Agent:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.load_state_dict(self.model.state_dict())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = SummaryWriter(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0

        self.terminate = False
        self.last_logged_episode = 0
        self.current_step = 0
        self.training_initialized = False
        self.optimizer = optim.AdamW(self.model.parameters(), lr=LEARNING_RATE, amsgrad=True)
        self.epsilon = EPSILON

    '''
    Creates the model for our Deep Q Learning agent.
    '''
    def create_model(self):
        # Using a CNN as our base model to process inputs
        model = decisionModel.UNetDecision(3,5).to(device)
        return model
    
    '''
    updates the replay memory.
    '''
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition) # Transition consists of [current_state, action, reward, new_state, done flag]

    '''
    Training process for our RL agent.
    '''
    def train_old(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Select the samples for our minibatch from the replay memory
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Grab the image from the minibatch items
        current_states = torch.tensor(np.array([transitions[0] for transitions in minibatch])/255).to(torch.float32) # Divide by 255 to have color values between 0 and 1 since our states are images
        current_qs_list = []
        future_qs_list = []
        for state in current_states:
            # Let the model choose an action
            state = state.unsqueeze(0).permute(0, 3, 1, 2)
            with torch.no_grad():
                current_qs_list.append(self.model(state))

        # Grab the new states from the minibatch items
        new_current_states = torch.tensor(np.array([transition[3] for transition in minibatch])/ 255).to(torch.float32) 
        for state in new_current_states:
            state = state.unsqueeze(0).permute(0, 3, 1, 2)
            # Let the target model choose the following action
            with torch.no_grad():
                future_qs_list.append(self.target_model(state))

        state_action_values = []
        expected_future_state_action_values = []

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                action_choices = future_qs_list[index]
                max_future_q = action_choices.max(dim=1)[0]
                new_q = reward + DISCOUNT_FACTOR * max_future_q
            else:
                # If we have crashed, we have no future rewards.
                new_q = reward

            #update our qs
            current_qs = current_qs_list[index][0]
            current_qs[action] = new_q[0]

            state_action_values.append(torch.tensor(current_state))
            expected_future_state_action_values.append(current_qs)
            
        # Log the reward for this step.
        log_this_step = False
        if self.current_step > self.last_logged_episode:
            log_this_step = True
            self.tensorboard.add_scalar("Reward", new_q, self.current_step)
            self.last_log_episode = self.current_step

        # Compute Loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_future_state_action_values)

        # Optimize model
        self.optimizer.zero_grad()
        # Update weights
        self.optimizer.step()

        if log_this_step:
            self.target_update_counter += 1

        #Update the target network every n steps
        if self.target_update_counter > UPDATE_TARGET_EVERY_N_STEPS:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update_counter = 0

        self.current_step += 1
    
    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Select the samples for our minibatch from the replay memory
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        # Transform batch to tensors and permute states aka camera images in shape [B, C, H, W]
        state_batch = torch.tensor(np.array([batchitem[0] for batchitem in minibatch])).to(torch.float32).permute(0, 3, 1, 2).to(device)
        action_batch = torch.tensor(np.array([batchitem[1] for batchitem in minibatch])).to(torch.int64).to(device)
        reward_batch = torch.tensor(np.array([batchitem[2] for batchitem in minibatch])).to(device)
        next_state_batch = torch.tensor(np.array([batchitem[3] for batchitem in minibatch])).to(torch.float32).permute(0, 3, 1, 2).to(device)
        done_mask = torch.tensor(np.array([0 if batchitem[4] == True else 1 for batchitem in minibatch])).to(torch.float32).to(device)
        # Compute Q(s_t, a) for the model
        state_action_values = self.model(state_batch)
        state_action_values = state_action_values.gather(1, action_batch.unsqueeze(1))

        next_state_values = torch.zeros(MINIBATCH_SIZE, device=device)

        # Compute V(s_{t+1})
        with torch.no_grad():
            next_state_values = (self.target_model(next_state_batch).max(1))[0]

        expected_state_action_values = reward_batch + DISCOUNT_FACTOR * next_state_values * done_mask

        # Compute Loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize model
        self.optimizer.zero_grad()
        # Update weights
        self.optimizer.step()

        if self.current_step > self.last_logged_episode:
            self.target_update_counter += 1

        #Update the target network every n steps
        if self.target_update_counter > UPDATE_TARGET_EVERY_N_STEPS:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update_counter = 0

        self.current_step += 1
        
        return




    def get_qs(self, state):
        return self.model(np.array(state).reshape(-1, *state.shape)/255)[0]

    def train_in_loop(self):

        self.training_initialized = True
        i = 1
        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)
            i = i+1