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
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.models import Model
from collections import deque
import tensorboard

REPLAY_MEMORY_SIZE = 5000
MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 32
PREDICTION_BATCH_SIZE = 1
DISCOUNT_FACTOR = 0.99
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY_N_STEPS = 5
MODEL_NAME = "Xception"

class DQNAgent_Xception:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = tensorboard.ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0

        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

    '''
    Creates the model for our Deep Q Learning agent.
    '''
    def create_model(self):
        # Using Xception as our base model to process inputs
        base_model = Xception(weights= None, include_top=False, input_shape=(settings.image_h, settings.image_w))
        # Add an avgPool layer
        x = GlobalAveragePooling2D()(base_model.output)
        # Add a 5 neuron activation layer to handle the inputs.
        predictions = Dense(5, activation="linear")(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(loss="mse", optimizer=Adam(lr=0.005), metrics=["accuracy"])
        return model
    
    '''
    updates the replay memory.
    '''
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition) # Transition consists of current_state, action, reward, new_state, and the done flag)

    '''
    Training process for our RL agent.
    '''
    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Select the samples for our minibatch from the replay memory
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transitions[0] for transitions in minibatch])/255 # Divide by 255 to have color values between 0 and 1
        
        #TODO: PREDICT WITH THE ACUTAL MODEL
        

        new_current_states = np.array([transition[3] for transition in minibatch]) / 255
        
        #TODO: PREDICT WITH THE TARGET MODEL
        

        X = []
        y = []

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT_FACTOR * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard.step

        #TODO: ADJUST FIT
        

        if log_this_step:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY_N_STEPS:
            #TODO: Change this...?
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

    def train_in_loop(self):
        X = np.random.uniform(size=(1, settings.image_h, settings.image_w, 3)).astype(np.float32)
        y = np.random.uniform(size=(1, 3)).astype(np.float32)
        with self.graph.as_default():
            self.model.fit(X,y, verbose=False, batch_size=1)

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)