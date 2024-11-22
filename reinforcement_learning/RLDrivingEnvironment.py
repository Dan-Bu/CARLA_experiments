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
MIN_REWARD = -10
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
    lane_invasion_history = []
    lane_invasion_count = 0
    lane_invasion_last_infraction_time = time.time()

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
        # Reset the collision history, lane invasion history and the actor list for a new run
        self.collision_history = []
        self.lane_invasion_history = []
        self.lane_invasion_count = 0
        self.actor_list = []

        # Get a new random spawn point, and spawn our vehicle there.
        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.vehicle_blueprint, self.transform)
        self.actor_list.append(self.vehicle)

        # ------------RGB CAM----------------
        # Configure the RGB camera
        CAMERA_POS_Z = 1.5
        CAMERA_POS_X = 0.5
        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.camera_bp.set_attribute('image_size_x', f'{self.image_width}') # this ratio works in CARLA 9.14 on Windows
        self.camera_bp.set_attribute('image_size_y', f'{self.image_height}')
        self.camera_init_trans = carla.Transform(carla.Location(z=CAMERA_POS_Z,x=CAMERA_POS_X))

        # this creates the camera in the sim
        self.front_camera_feed = self.world.spawn_actor(self.camera_bp, self.camera_init_trans, attach_to=self.vehicle)
        self.actor_list.append(self.front_camera_feed)
        
        # Set up listening to the camera
        self.front_camera_feed.listen(lambda data: self.process_rgb_img(data))

        # ------------SEMSEG CAM----------------
        # Configure the SemSeg camera
        CAMERA_POS_Z = 1.5
        CAMERA_POS_X = 0.5
        self.semseg_camera_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        self.semseg_camera_bp.set_attribute('image_size_x', f'{self.image_width}') # this ratio works in CARLA 9.14 on Windows
        self.semseg_camera_bp.set_attribute('image_size_y', f'{self.image_height}')
        self.semseg_camera_init_trans = carla.Transform(carla.Location(z=CAMERA_POS_Z,x=CAMERA_POS_X))

        # this creates the Semsegcamera in the sim
        self.semseg_front_camera_feed = self.world.spawn_actor(self.semseg_camera_bp, self.semseg_camera_init_trans, attach_to=self.vehicle)
        self.actor_list.append(self.semseg_front_camera_feed)
        
        # Set up listening to the semseg camera
        self.semseg_front_camera_feed.listen(lambda data: self.process_semseg_img(data))

        # initialise our brake and throttle as 0 to not get weird inputs at the start.
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        time.sleep(3) # sleep for a moment in order to avoid a collision caused by the initial spawn fall.

        # Generate our collision sensor
        collision_sensor = self.world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_sensor, self.camera_init_trans, attach_to=self.vehicle)
        self.actor_list.append(self.collision_sensor)
        self.collision_sensor.listen(lambda event: self.collision_data(event))

        # Generate our lane invasion sensor
        lane_invasion_sensor = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.collision_sensor = self.world.spawn_actor(lane_invasion_sensor, self.camera_init_trans, attach_to=self.vehicle)
        self.actor_list.append(self.collision_sensor)
        self.collision_sensor.listen(lambda event: self.lane_invasion_data(event))

        # Wait for the camera feed to start, in case it takes longer than the 3 seconds to initialize.
        while self.front_camera_feed is None:
            time.sleep(0.01)

        # Log the time when the sensors are initiated as the actual start time to cut away the falling time from the episode
        self.episode_start = time.time()

        # Make sure brake and throttle are at 0 to begin the episode
        self.vehicle.apply_control(carla.VehicleControl(brake=0.0, throttle=0.0))

        return self.semseg_front_camera_feed
    
    '''
    Step function that takes the action that is taken, and returns the observation, reward, 
    done flag, and extra info if we need it.
    '''
    def step(self, action):

        # Compute the current speed
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        # Roughly regulate speed to constant 50
        if kmh > 30:
            throttle_input = 0.0
        else:
            throttle_input = 1.0

        # Keep actions simple to start off with
        if action == 0: # Go full left
            self.vehicle.apply_control(carla.VehicleControl(throttle=throttle_input, steer=-1.0*self.steer_amount))
        elif action == 1: # Go three quarter left
            self.vehicle.apply_control(carla.VehicleControl(throttle=throttle_input, steer=-0.75*self.steer_amount))
        elif action == 2: # Go half left
            self.vehicle.apply_control(carla.VehicleControl(throttle=throttle_input, steer=-0.5*self.steer_amount))
        elif action == 3: # Go one quarter left
            self.vehicle.apply_control(carla.VehicleControl(throttle=throttle_input, steer=-0.25*self.steer_amount))
        elif action == 4: # Go Straight
            self.vehicle.apply_control(carla.VehicleControl(throttle=throttle_input, steer=0))
        elif action == 5: # Go one quarter right
            self.vehicle.apply_control(carla.VehicleControl(throttle=throttle_input, steer=0.25*self.steer_amount))
        elif action == 6: # Go half right
            self.vehicle.apply_control(carla.VehicleControl(throttle=throttle_input, steer=0.5*self.steer_amount))
        elif action == 7: # Go three quarter right
            self.vehicle.apply_control(carla.VehicleControl(throttle=throttle_input, steer=0.75*self.steer_amount))
        elif action == 8: # Go full right
            self.vehicle.apply_control(carla.VehicleControl(throttle=throttle_input, steer=1.0*self.steer_amount))

        # Penalize crashes, as well as being slower than 50kmh. Reward clean driving
        if len(self.collision_history) > 0:
            done = True
            reward = -20
        else:
            done = False
            reward = 1

        # Punish leaving the lane (and therefore the road). Only one infraction per second to avoid getting multiple while crossing a line once.
        current_time = time.time()
        if len(self.lane_invasion_history) > self.lane_invasion_count and current_time - self.lane_invasion_last_infraction_time > 1:
            self.lane_invasion_count = len(self.lane_invasion_history)
            reward -= 5
            self.lane_invasion_last_infraction_time = current_time
        #set the done flag if the time for this episode is over
        if self.episode_start + EPISODE_TIME_LIMIT < time.time():
            done = True

        # return our observation (camera), reward, done flag, and any extra info
        return self.semseg_front_camera_feed, reward, done, None

    '''
    Method to display the camera image
    '''
    def process_rgb_img(self, image):
        # Convert image to array
        image_arr = np.array(image.raw_data)  
        # Bring it in RGBA shape
        image_arr = image_arr.reshape((self.image_height, self.image_width, 4))
        # Discard alpha channel
        image_arr = image_arr[:, :, :3]

        if self.show_cam:
            cv2.imshow("rgb", image_arr)
            cv2.waitKey(1)
        self.front_camera_feed = image_arr

    def process_semseg_img(self, image):
        image.convert(carla.ColorConverter.CityScapesPalette)
        # Convert image to array
        image_arr = np.array(image.raw_data)  
        # Bring it in RGBA shape
        image_arr = image_arr.reshape((self.image_height, self.image_width, 4))
        # Discard alpha channel
        image_arr = image_arr[:, :, :3]
        
        if self.show_cam:
            cv2.imshow("semseg", image_arr)
            cv2.waitKey(1)
        self.semseg_front_camera_feed = image_arr

    # Method to add an event to the env's history
    def collision_data(self, event):
        self.collision_history.append(event)

    def lane_invasion_data(self, event):
        self.lane_invasion_history.append(event)

    