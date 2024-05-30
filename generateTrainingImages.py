# Modifying previous preps notebook code to generate
# training images
# Note: this generates some black only images - they need to be deleted before training the model
# you can sort files by size - they will be all 3kb 
# Also you may need to run this with version 9.13 of Carla

#all imports
import carla #the sim library itself
from carla import command

import time # to set a delay after each photo
import carla.libcarla
import cv2 #to work with images from cameras
import numpy as np #in this example to change image representation - re-shaping
import math
import sys
import random
import os
sys.path.append('C:/Users/Daniel/Documents/PrivateProjects/CARLA/CARLA_0.9.15/WindowsNoEditor/PythonAPI/carla') # tweak to where you put carla
from agents.navigation.global_route_planner import GlobalRoutePlanner

import settings
import generateTraffic

SPEED_THRESHOLD = 2 #defines when we get close to desired speed so we drop the throttle

# Max steering angle
MAX_STEER_DEGREES = 40
#Max spin angle
MAX_SPIN = 20
# This is max actual angle with Mini under steering input=1.0
STEERING_CONVERSION = 75

#camera mount offset on the car - this mimics Tesla Model 3 view 
CAMERA_POS_Z = 1.5
CAMERA_POS_X = 0.5 

# Camera Callback. Defines what the camera should do when a frame is generated.
def camera_callback(image, data_dictionary):
    #Define the offsets to cut the image
    data_dictionary['rgb_image'] = np.reshape(np.copy(image.raw_data),(image.height,image.width,4))

# Callback for semantic segmentation camera
def sem_callback(image,data_dictionary):
    ########## IMPORTANT CHANGE for Semantic camera ##############
    image.convert(carla.ColorConverter.CityScapesPalette)
    data_dictionary['sem_image'] = np.reshape(np.copy(image.raw_data),(image.height,image.width,4))

# function to get angle between the car and target waypoint
def get_angle(car,wp):
    '''
    this function returns degrees between the car's direction 
    and direction to a selected waypoint
    '''
    vehicle_pos = car.get_transform()
    car_x = vehicle_pos.location.x
    car_y = vehicle_pos.location.y
    wp_x = wp.transform.location.x
    wp_y = wp.transform.location.y
    
    # vector to waypoint
    x = (wp_x - car_x)/((wp_y - car_y)**2 + (wp_x - car_x)**2)**0.5
    y = (wp_y - car_y)/((wp_y - car_y)**2 + (wp_x - car_x)**2)**0.5
    
    #car vector
    car_vector = vehicle_pos.get_forward_vector()
    degrees = math.degrees(np.arctan2(y, x) - np.arctan2(car_vector.y, car_vector.x))
    # extra checks on predicted angle when values close to 360 degrees are returned
    if degrees<-180:
        degrees = degrees + 360
    elif degrees > 180:
        degrees = degrees - 360
    return degrees

def get_proper_angle(car,wp_idx,rte):
    '''
    This function uses simple fuction above to get angle but for current
    waypoint and a few more next waypoints to ensure we have not skipped
    next waypoint so we avoid the car trying to turn back
    '''
    # create a list of angles to next 5 waypoints starting with current
    next_angle_list = []
    for i in range(10):
        if wp_idx + i*3 <len(rte)-1:
            next_angle_list.append(get_angle(car,rte[wp_idx + i*3][0]))
    idx = 0
    while idx<len(next_angle_list)-2 and abs(next_angle_list[idx])>40:
        idx +=1
    return wp_idx+idx*3,next_angle_list[idx]  

def get_distant_angle(car,wp_idx,rte, delta):
    '''
    This function modifies the function above to get angle to a waypoint
    at a distance so we could use it for training image generation
    
    We will display the angle for now in the 'telemetry' view so
    we could play with how far forward we need to pick the waypoint
    '''
    if wp_idx + delta < len(rte)-1:
        i = wp_idx + delta
    else:
        i = len(rte)-1
    # check for intersection within the "look forward"
    # so we do not give turn results when just following the road
    intersection_detected = False
    for x in range(i-wp_idx):
        if rte[wp_idx+x][0].is_junction:
             intersection_detected = True
    angle = get_angle(car,rte[i][0])
    if not intersection_detected:
        result = 0
    elif angle <-10:
        result = -1
    elif angle>10:
        result =1
    else:
        result = 0    
    return result

def draw_route(world, wp, route,seconds=3.0):
    #draw the next few points route in sim window - Note it does not
    # get into the camera of the car
    if len(route)-wp <25: # route within 25 points from end is red
        draw_colour = carla.Color(r=255, g=0, b=0)
    else:
        draw_colour = carla.Color(r=0, g=0, b=255)
    for i in range(10):
        if wp+i<len(route)-2:
            world.debug.draw_string(route[wp+i][0].transform.location, '^', draw_shadow=False,
                color=draw_colour, life_time=seconds,
                persistent_lines=True)
    return None


def select_random_route(world, position,locs):
    '''
    retruns a random route for the car/veh
    out of the list of possible locations locs
    where distance is longer than 100 waypoints
    '''    
    point_a = position.location #we start at where the car is or last waypoint
    sampling_resolution = 1
    grp = GlobalRoutePlanner(world.get_map(), sampling_resolution)
    # now let' pick the longest possible route
    min_distance = 100
    result_route = None
    route_list = []
    for loc in locs: # we start trying all spawn points 
                                #but we just exclude first at zero index
        cur_route = grp.trace_route(point_a, loc.location)
        if len(cur_route) > min_distance:
            route_list.append(cur_route)
    result_route = random.choice(route_list)
    return result_route

def find_highest_number_in_filenames(folder_path):
    highest_number = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            try:
                number = int(filename.split('.')[0])
                if number > highest_number:
                    highest_number = number
            except ValueError:
                continue
    return highest_number

def generate_training_data():
    #set up data dict

    camera_data = {'sem_image': np.zeros((settings.image_h,settings.image_w,4)),
                   'rgb_image': np.zeros((settings.image_h,settings.image_w,4))}

    # connect to the sim 
    client = carla.Client('localhost', 2000)

    #time.sleep(5)
    #client.set_timeout(25)
    maps =  ["Town01", "Town02", "Town03", "Town04", "Town05", "Town06", "Town07"]

    #client.load_world(maps[0], reset_settings=False, map_layers=carla.MapLayer.All)

    test_data_offset = find_highest_number_in_filenames("./out/semseg_val/images") + 1

    # get world and spawn points
    world = client.get_world()

    # ensure sync mode on 
    if settings.images_from_fixed_routes:
        world_settings = world.get_settings()
        world_settings.synchronous_mode = True
        world_settings.fixed_delta_seconds = 0.1
        world_settings.no_rendering_mode = False
        world.apply_settings(world_settings)

    spawn_points = world.get_map().get_spawn_points()

    #clean up any existing cars
    for actor in world.get_actors().filter('*vehicle*'):
        actor.destroy()
    for actor in world.get_actors().filter('*walker*'):
        actor.destroy()

    

   
    

    def exit_clean():
        #clean up
        cv2.destroyAllWindows()
        for sensor in world.get_actors().filter('*sensor*'):
            sensor.destroy()
        for actor in world.get_actors().filter('*vehicle*'):
            actor.destroy()
        for actor in world.get_actors().filter('*walker*'):
            actor.destroy()
        return None

    exit_clean()


    # main loop
    img_counter = 0
    quit = False
    training_data_dict = [] # Dictionary that holds the metadata used for training.
    iteration_num = 1

    def per_cycle_setup():
        # Load a random map
        world.unload_map_layer(carla.MapLayer.All) #Required to unload navigation data from previous map
        map_index = random.randint(0,len(maps)-1)
        map_index = 2       
        success = False
        while success == False:
            try:
                client.load_world_if_different(maps[map_index], reset_settings=False, map_layers=carla.MapLayer.All)
                success = True

            except RuntimeError:
                print(f"Failed to load world with index {map_index}")
                success = False
        
        weather = world.get_weather()
        # Set up random weather
        weather.wind_strength = random.randint(0,100)
        if random.randint(0,3) == 3:    # Nighttime?
            weather.sun_altitude_angle = random.randint(181, 360) # Time of day
            weather.sun_azimuth_angle = 90
        else:                           # Daytime?
            weather.sun_altitude_angle = random.randint(20, 160) # Time of day. 90 is noon, 0 morning, 180 evening
            weather.sun_azimuth_angle = -90

        rain = False
        weather.precipitation = 0
        weather.precipation_deposits = 0
        weather.cloudiness = random.randint(0, 40)
        if random.randint(0,2) == 2:    # Check if we have rain
            weather.precipitation  = random.randint(0,100)
            weather.precipation_deposits = random.randint(0,100)
            weather.cloudiness = random.randint(50,100)
            weather.wetness = weather.precipitation
            rain = True
        else:
            if random.randint(0,1) == 1: # 50% of the time where it doesn't rain, we want puddles
                precipation_deposits = random.randint(0,20)
        
        weather.fog_density = 0
        weather.fog_distance = 0
        if rain == False:   # Only generate fog if we have no rain
            if random.randint(0,2) == 2:
                weather.fog_density = random.randint(0,100) # Thickness of the fog
                weather.fog_distance = random.randint(0,200) # View distance in fog
                weather.fog_falloff = 1/random.randint(0,100) # How heavy (i.e. low to the ground) the fog is
                weather.wetness = random.randint(0, 10) # Wetness of the scene

        world.set_weather(weather)

        # Determine spawn point for the ego vehicle
#        spawn_points = world.get_map().get_spawn_points()
        start_point = random.choice(spawn_points)

        #look for a blueprint of dodge charger car
        vehicle_bp = world.get_blueprint_library().find('vehicle.dodge.charger_2020')
        vehicle_bp.set_attribute('role_name', 'hero')
        ego_vehicle = None
        while ego_vehicle == None:
            ego_vehicle = world.try_spawn_actor(vehicle_bp, start_point)

        world.tick()
        time.sleep(2)        
        #Set lights to on if it is night, otherwise turn off
        light_state = ego_vehicle.get_light_state()
        if weather.sun_altitude_angle < 10:
            light_state = carla.libcarla.VehicleLightState.LowBeam

        else:
            light_state = carla.libcarla.VehicleLightState.NONE

        #command.SetVehicleLightState(ego_vehicle, light_state) #This is broken for some reason?
        ego_vehicle.set_light_state(light_state)
        #---Create Spectator---
        spectator_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        spectator_transform = carla.Transform(carla.Location(x=-5, z=2), carla.Rotation(pitch=-15))

        spectator = world.spawn_actor(spectator_bp, spectator_transform, attach_to=ego_vehicle)

        return ego_vehicle, start_point, spectator_transform, spectator
    
    def sensor_setup():
        # setting semantic camera
        camera_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        camera_bp.set_attribute('image_size_x', str(settings.image_w)) # this ratio works in CARLA 9.13 on Windows
        camera_bp.set_attribute('image_size_y', str(settings.image_h))
        camera_bp.set_attribute('fov', '90')
        camera_init_trans = carla.Transform(carla.Location(z=CAMERA_POS_Z,x=CAMERA_POS_X))
        camera_sem = world.spawn_actor(camera_bp,camera_init_trans,attach_to=ego_vehicle)
        # this actually opens a live stream from the sem_camera
        camera_sem.listen(lambda image: sem_callback(image,camera_data))


        # setting RGB Camera 
        camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(settings.image_w)) 
        camera_bp.set_attribute('image_size_y', str(settings.image_h))
        camera_init_trans = carla.Transform(carla.Location(z=CAMERA_POS_Z,x=CAMERA_POS_X))
        # this creates the camera in the sim
        camera = world.spawn_actor(camera_bp,camera_init_trans,attach_to=ego_vehicle)
        image_w = camera_bp.get_attribute('image_size_x').as_int()
        image_h = camera_bp.get_attribute('image_size_y').as_int()

        # this actually opens a live stream from the camera
        camera.listen(lambda image: camera_callback(image,camera_data))

        #---Advance the world tick---
        world.tick()
        return camera, camera_sem

       

        

    if settings.images_from_fixed_routes:
        while img_counter < settings.amount_training_data_to_generate:
            
            ego_vehicle, start_point, spectator_transform, spectator = per_cycle_setup()
            camera, camera_sem = sensor_setup()
            
            prev_position = ego_vehicle.get_transform()
            #---getting a random route for the car---
            route = select_random_route(world, start_point,spawn_points)
            curr_wp = 0
            quit = False
             #---Create the CV2 windows---
            cv2.namedWindow('RGB Camera',cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RGB Camera',camera_data['rgb_image'])

            while curr_wp<len(route)-10 and img_counter < 1000 * iteration_num and img_counter < settings.amount_training_data_to_generate:
                # move the car to next waypoint
                curr_wp +=1
                ego_vehicle.set_transform(route[curr_wp][0].transform)
                #time.sleep(2)
                world.tick()

                # Update the camera's location and rotation to match the ego_vehicle
                ego_transform = ego_vehicle.get_transform()
                spectator_transform.location = ego_transform.location + carla.Location(x=-5, z=2)
                spectator_transform.rotation = ego_transform.rotation + carla.Rotation(pitch=-15)

                # Set the camera's new transform
                spectator.set_transform(spectator_transform)

                # first position the car on the route and take normal shot (without spinning)
                _, predicted_angle = get_proper_angle(ego_vehicle,curr_wp+5,route)
                steer_input = predicted_angle
                # limit steering to max angel, say 40 degrees
                if predicted_angle<-MAX_STEER_DEGREES:
                    steer_input = -MAX_STEER_DEGREES
                elif predicted_angle>MAX_STEER_DEGREES:
                    steer_input = MAX_STEER_DEGREES
                gen_dir_angle = get_distant_angle(ego_vehicle,curr_wp,route,30)
                initial_yaw = ego_vehicle.get_transform().rotation.yaw
                sem_im = camera_data['sem_image']
                image = camera_data['rgb_image']
                #img_counter += 1
                time_grab = time.time_ns()
                cv2.waitKey(10)
                cv2.imwrite(f'{settings.sem_seg_data_path}/masks/{str(img_counter)}.png', sem_im)
                cv2.imwrite(f'{settings.sem_seg_data_path}/images/{str(img_counter)}.png',  image)
                training_data_dict.append({"index": img_counter, "steering_angle": round(steer_input, 2), "car_yaw": ego_vehicle.get_transform().rotation.yaw})
                image = cv2.putText(image, 'Steer: '+str(steer_input), settings.text_loc1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                image = cv2.putText(image, 'yaw: '+str(ego_vehicle.get_transform().rotation.yaw), settings.text_loc2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                image = cv2.putText(image, 'image number: '+str(img_counter), settings.text_loc3, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow('RGB Camera', image)
                if cv2.waitKey(0) == ord('q'):
                            quit = True
                            break
            
            iteration_num += 1
            exit_clean()    #Clean up after each individual run
            if quit:
                break

    else:
        while img_counter < settings.amount_training_data_to_generate:

            ego_vehicle, start_point, spectator_transform, spectator = per_cycle_setup()
            camera, camera_sem = sensor_setup()
            vehicles_list, walkers_list, all_id, batch = generateTraffic.generate_traffic(client, 40, 20)

            world.tick()
            time.sleep(2)        
            
            # Set autopilot on for all vehicles so they drive around.
            tm = client.get_trafficmanager(8000)
            tm_port = tm.get_port()
            for vehicle in world.get_actors().filter('*vehicle*'):
                vehicle.set_autopilot(True, tm_port)

            ego_vehicle.set_autopilot(True, tm_port)
             #---Create the CV2 windows---
            cv2.namedWindow('RGB Camera',cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RGB Camera',camera_data['rgb_image'])
            quit = False
            prev_position = ego_vehicle.get_transform().location
            tick_counter = 0
            while img_counter < settings.amount_training_data_to_generate:
                world.tick()
                sem_im = camera_data['sem_image']
                image = camera_data['rgb_image']
                
                time_grab = time.time_ns()
                cv2.waitKey(10)
                ego_transform = ego_vehicle.get_transform()
                tick_counter+=1
                if abs(ego_transform.location.x - prev_position.x) + abs(ego_transform.location.y - prev_position.y) + abs(ego_transform.location.z - prev_position.z) > 10.0:    # only take a picture after a certain amount of movement
                    cv2.imwrite(f'{settings.sem_seg_data_path}/masks/{str(test_data_offset + img_counter)}.png', sem_im)
                    cv2.imwrite(f'{settings.sem_seg_data_path}/images/{str(test_data_offset + img_counter)}.png',  image)
                    img_counter += 1
                    prev_position = ego_transform.location
                    tick_counter = 0
                elif tick_counter > 20000:
                    break
                training_data_dict.append({"index": img_counter, "steering_angle": round(0, 2), "car_yaw": ego_vehicle.get_transform().rotation.yaw})
                image = cv2.putText(image, f'x: {str(ego_vehicle.get_transform().location.x)}, y: {str(ego_vehicle.get_transform().location.y)}, z: {str(ego_vehicle.get_transform().location.z)}', settings.text_loc1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                image = cv2.putText(image, 'yaw: '+str(ego_vehicle.get_transform().rotation.yaw), settings.text_loc2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                image = cv2.putText(image, 'image number: '+str(img_counter), settings.text_loc3, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                cv2.imshow('RGB Camera', image)
                
                if cv2.waitKey(10) == ord('q'):
                            quit = True
                            break
                            
            #exit_clean()    #Clean up after each individual run
            tm.shut_down()
            iteration_num +=1
            if quit:
                break        
    return
