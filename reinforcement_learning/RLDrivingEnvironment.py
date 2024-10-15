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


# Constants
SHOW_VIDEO = False
EPISODE_TIME_LIMIT = 20
'''
Reinforcement learning environment for a vehicle in Carla. Implements Q-Learning 
'''
class vehicleEnv:
    show_cam = SHOW_VIDEO
    steer_amount = 1.0
    image_width = settings.image_w
    image_height = settings.image_h
    actor_list = []
    front_camera_feed = None
    collision_history = []

    def __init__(self):
        # initialise the world
        self.client = carla.Client('localhost', 2000)
        self.client.load_world("Town02", reset_settings=False, map_layers=carla.MapLayer.All)
        self.world = self.client.get_world()

        # Get out vehicle Blueprint
        self.vehicle_blueprint = self.world.get_blueprint_library().find('vehicle.dodge.charger_2020')

    '''
    Resets the vehicle state
    '''
    def reset(self):
        # Reset the collision history and the actor list for a new run
        self.collision_hist = []
        self.actor_list = []

        # Get a new random spawn point, and spawn our vehicle there.
        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.vehicle_blueprint, self.transform)
        self.actor_list.append(self.vehicle)

        # Configure the RGB camera
        self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')
        CAMERA_POS_Z = 1.5
        CAMERA_POS_X = 0.5
        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.camera_bp.set_attribute('image_size_x', f'{self.image_width}') # this ratio works in CARLA 9.14 on Windows
        self.camera_bp.set_attribute('image_size_y', f'{self.image_height}')
        self.camera_init_trans = carla.Transform(carla.Location(z=CAMERA_POS_Z,x=CAMERA_POS_X))

        # this creates the camera in the sim
        self.front_camera = self.world.spawn_actor(self.camera_bp, self.camera_init_trans, attach_to=self.vehicle)
        self.actor_list.append(self.front_camera)
        # Set up listening to the camera
        self.front_camera.listen(lambda data: self.process_img(data))

        # initialise our brake and throttle as 0 to not get weird inputs at the start.
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        time.sleep(3) # sleep for a moment in order to avoid a collision caused by the initial spawn fall.

        # Generate our collision sensor
        collision_sensor = self.world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_sensor, self.camera_init_trans, attach_to=self.vehicle)
        self.actor_list.append(self.collision_sensor)

        # Wait for the camera feed to start, in case it takes longer than the 3 seconds to initialize.
        while self.front_camera_feed is None:
            time.sleep(0.01)

        # Log the time when the sensors are initiated as the actual start time to cut away the falling time from the episode
        self.episode_start = time.time

        # Make sure brake and throttle are at 0 to begin the episode
        self.vehicle.apply_control(carla.VehicleControl(brake=0.0, throttle=0.0))

        return self.front_camera
    
    '''
    Step function that takes the action that is taken, and returns the observation, reward, 
    done flag, and extra info if we need it.
    '''
    def step(self, action):
        # Keep actions simple to start off with
        if action == 0: # Go half left
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-0.5*self.steer_amount))
        elif action == 1: # Go full left
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1.0*self.steer_amount))
        elif action == 2: # Go Straight
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))
        
        elif action == 3: # Go half right
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.5*self.steer_amount))
        elif action == 4: # Go full right
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1.0*self.steer_amount))

        # Compute the current speed
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        # Penalize crashes, as well as being slower than 50kmh. Reward clean driving
        if len(self.collision_hist) > 0:
            done = True
            reward = -200
        elif kmh < 50:
            done = False
            reward = -1
        else:
            done = False
            reward = 1

        #set the done flag if the time for this episode is over
        if self.episode_start + EPISODE_TIME_LIMIT < time.time():
            done = True

        # return our observation (camera), reward, done flag, and any extra info
        return self.front_camera, reward, done, None

    '''
    Method to display the camera image
    '''
    def process_img(self, image):
        # Convert image to array
        image_arr = np.array(image.raw_data)  
        # Bring it in RGBA shape
        image_arr = image_arr.reshape((self.image_height, self.image_width, 4))
        # Discard alpha channel
        image_arr = image_arr[:, :, :3]

        if self.show_cam:
            cv2.imshow("", image_arr)
            cv2.waitKey(1)
        self.front_camera = image_arr

    # Method to add an event to the env's history
    def collision_data(self, event):
        self.collision_hist.append(event)

    