
from carla import command
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from agents.navigation.global_route_planner import GlobalRoutePlanner

import reinforcement_learning.decisionModel as decisionModel

REPLAY_MEMORY_SIZE = 5000
MIN_REPLAY_MEMORY_SIZE = 500
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
DISCOUNT_FACTOR = 0.99
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 8
UPDATE_TARGET_EVERY_N_STEPS = 5
MODEL_NAME = "ConvDecisionV2"
MODEL_OUTPUT_SIZE = 9
NUM_EPISODES = 2001
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
        model = None
        if MODEL_NAME == "ConvDecision":
            model = decisionModel.ConvDecision(3,MODEL_OUTPUT_SIZE).to(device)
        elif MODEL_NAME == "ConvDecisionV2":
            model = decisionModel.ConvDecisionV2(3,MODEL_OUTPUT_SIZE).to(device)
        else:
            print("Model not supported!")
            quit()
        return model
    
    '''
    updates the replay memory.
    '''
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition) # Transition consists of [current_state, action, reward, new_state, done flag]


    '''
    Training process for our RL agent.
    '''    
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
        # Set gradients to zero
        self.optimizer.zero_grad()
        
        # Compute Loss through backprop
        loss.backward()
        
        # Update weights with new gradient
        self.optimizer.step()

        if self.current_step > self.last_logged_episode:
            self.tensorboard.add_scalar("Reward", max(next_state_values), self.current_step)
            #self.tensorboard.add_scalar("Loss", loss[0], self.current_step)
            self.target_update_counter += 1

        #Update the target network every n steps
        if self.target_update_counter > UPDATE_TARGET_EVERY_N_STEPS:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update_counter = 0

        self.current_step += 1
        
        return




    def get_qs(self, state):
        tensor = torch.tensor(np.array(state).reshape(-1, *state.shape)/255)[0].unsqueeze(0).permute(0,3,1,2).to(torch.float32).to(device)
        return self.model(tensor)

    def train_in_loop(self):

        self.training_initialized = True
        i = 1
        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)
            i = i+1




