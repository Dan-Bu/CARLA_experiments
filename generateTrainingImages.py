# Modifying previous preps notebook code to generate
# training images
# Note: this generates some black only images - they need to be deleted before training the model
# you can sort files by size - they will be all 3kb 
# Also you may need to run this with version 9.13 of Carla

#all imports
import carla #the sim library itself
import time # to set a delay after each photo
import cv2 #to work with images from cameras
import numpy as np #in this example to change image representation - re-shaping
import math
import sys
import random
sys.path.append('C:/Users/Daniel/Documents/PrivateProjects/CARLA/CARLA_0.9.15/WindowsNoEditor/PythonAPI/carla') # tweak to where you put carla
from agents.navigation.global_route_planner import GlobalRoutePlanner

import settings

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



def generate_training_data():
    #set up data dict

    camera_data = {'sem_image': np.zeros((settings.image_h,settings.image_w,4)),
                   'rgb_image': np.zeros((settings.image_h,settings.image_w,4))}

    # connect to the sim 
    client = carla.Client('localhost', 2000)

    #time.sleep(5)
    #client.set_timeout(25)
    client.load_world('Town03', reset_settings=False, map_layers=carla.MapLayer.All)


    # get world and spawn points
    world = client.get_world()

    # ensure sync mode on 
    world_settings = world.get_settings()
    world_settings.synchronous_mode = True
    world_settings.fixed_delta_seconds = 0.1
    world_settings.no_rendering_mode = False
    world.apply_settings(world_settings)

    spawn_points = world.get_map().get_spawn_points()

    #clean up any existing cars
    for actor in world.get_actors().filter('*vehicle*'):
        actor.destroy()

    #look for a blueprint of dodge charger car
    vehicle_bp = world.get_blueprint_library().find('vehicle.dodge.charger_2020')
    vehicle_bp.set_attribute('role_name', 'hero')

   
    

    def exit_clean():
        #clean up
        cv2.destroyAllWindows()
        camera_sem.stop()
        for sensor in world.get_actors().filter('*sensor*'):
            sensor.destroy()
        for actor in world.get_actors().filter('*vehicle*'):
            actor.destroy()
        return None

    


    #main loop
    img_counter = 0
    quit = False
    training_data_dict = [] # Dictionary that holds the metadata used for training.
    while img_counter < settings.amount_training_data_to_generate:

        start_point = random.choice(spawn_points)
        start_point = spawn_points[1]
        ego_vehicle = world.try_spawn_actor(vehicle_bp, start_point)

        time.sleep(2)

        # setting semantic camera
        camera_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        camera_bp.set_attribute('image_size_x', str(settings.image_w)) # this ratio works in CARLA 9.13 on Windows
        camera_bp.set_attribute('image_size_y', str(settings.image_h))
        camera_bp.set_attribute('fov', '90')
        camera_init_trans = carla.Transform(carla.Location(z=CAMERA_POS_Z,x=CAMERA_POS_X))
        camera_sem = world.spawn_actor(camera_bp,camera_init_trans,attach_to=ego_vehicle)
        # this actually opens a live stream from the sem_camera
        camera_sem.listen(lambda image: sem_callback(image,camera_data))


        #setting RGB Camera 
        camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(settings.image_w)) 
        camera_bp.set_attribute('image_size_y', str(settings.image_h))
        camera_init_trans = carla.Transform(carla.Location(z=CAMERA_POS_Z,x=CAMERA_POS_X))
        #this creates the camera in the sim
        camera = world.spawn_actor(camera_bp,camera_init_trans,attach_to=ego_vehicle)
        image_w = camera_bp.get_attribute('image_size_x').as_int()
        image_h = camera_bp.get_attribute('image_size_y').as_int()

        # this actually opens a live stream from the camera
        camera.listen(lambda image: camera_callback(image,camera_data))

        #---Set Autopilot---
        #tm = client.get_trafficmanager(8000)
        #tm_port = tm.get_port()
        #ego_vehicle.set_autopilot(True, tm_port)


        #---Advance the world tick---
        world.tick()

        #---Create the CV2 windows---
        cv2.namedWindow('RGB Camera',cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RGB Camera',camera_data['rgb_image'])


        prev_position = ego_vehicle.get_transform()
        #---getting a random route for the car---
        route = select_random_route(world, start_point,spawn_points)
        curr_wp = 0

        while curr_wp<len(route)-6 and img_counter < settings.amount_training_data_to_generate:
            # move the car to next waypoint
            curr_wp +=1
            ego_vehicle.set_transform(route[curr_wp][0].transform)
            #time.sleep(2)
            world.tick()

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
            #cv2.imwrite(f'C:/Users/Daniel/Documents/PrivateProjects/CARLA/Code/out/semseg1/{str(img_counter)}.png', sem_im)
            #cv2.imwrite(f'C:/Users/Daniel/Documents/PrivateProjects/CARLA/Code/out/camera1/{str(img_counter)}.png',  image)
            training_data_dict.append({"index": img_counter, "steering_angle": round(steer_input, 2), "car_yaw": ego_vehicle.get_transform().rotation.yaw})
            image = cv2.putText(image, 'Steer: '+str(steer_input), settings.org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            image = cv2.putText(image, 'yaw: '+str(ego_vehicle.get_transform().rotation.yaw), settings.org2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow('RGB Camera', image)

    return
    while img_counter < 100:
        start_point = random.choice(spawn_points)
        ego_vehicle = world.try_spawn_actor(vehicle_bp[0], start_point)
        time.sleep(2)
        # setting semantic camera
        camera_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        camera_bp.set_attribute('image_size_x', '640') # this ratio works in CARLA 9.13 on Windows
        camera_bp.set_attribute('image_size_y', '480')
        camera_bp.set_attribute('fov', '90')
        camera_init_trans = carla.Transform(carla.Location(z=CAMERA_POS_Z,x=CAMERA_POS_X))
        camera_sem = world.spawn_actor(camera_bp,camera_init_trans,attach_to=ego_vehicle)
        image_w = 640
        image_h = 480

        camera_data = {'sem_image': np.zeros((image_h,image_w,4)),
                       'rgb_image': np.zeros((image_h,image_w,4))}

            # this actually opens a live stream from the cameras
        camera_sem.listen(lambda image: sem_callback(image,camera_data))
        # adding collision sensor

        #setting RGB Camera 
        camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '640') 
        camera_bp.set_attribute('image_size_y', '360')
        camera_init_trans = carla.Transform(carla.Location(z=CAMERA_POS_Z,x=CAMERA_POS_X))
        #this creates the camera in the sim
        camera = world.spawn_actor(camera_bp,camera_init_trans,attach_to=ego_vehicle)
        image_w = camera_bp.get_attribute('image_size_x').as_int()
        image_h = camera_bp.get_attribute('image_size_y').as_int()
    
        # this actually opens a live stream from the camera
        camera.listen(lambda image: camera_callback(image,camera_data))
        cv2.namedWindow('RGB Camera',cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RGB Camera',camera_data['rgb_image'])



        prev_position = ego_vehicle.get_transform()
        # getting a random route for the car
        route = select_random_route(world, start_point,spawn_points)
        curr_wp = 0
        world.tick()

        while curr_wp<len(route)-6 and img_counter < 100:

            # move the car to next waypoint
            curr_wp +=1
            ego_vehicle.set_transform(route[curr_wp][0].transform)
            #time.sleep(2)
            world.tick()

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
            img_counter += 1
            time_grab = time.time_ns()
            cv2.imwrite('_img/%06d_%s_%s.png' % (time_grab, gen_dir_angle,round(steer_input,0)), sem_im)
            cv2.imshow('RGB Camera',image)
             #only ouside intersections we spin the car around
            if route[curr_wp][0].is_intersection==False and route[curr_wp][0].is_junction==False:
                #grab images while spinning the car around
                for i in range(3):
                    # Carla Tick
                    trans = ego_vehicle.get_transform()
                    angle_adj = random.randrange(-MAX_SPIN, MAX_SPIN, 1)
                    trans.rotation.yaw = initial_yaw +angle_adj 
                    ego_vehicle.set_transform(trans)
                    world.tick()
                    time.sleep(2)  #these delays seem to be necessary for the car to take the position before a shot is taken
                    steer_input = predicted_angle - angle_adj # we put the opposite to correct back to straight
                    if steer_input<-MAX_STEER_DEGREES:
                        steer_input = -MAX_STEER_DEGREES
                    elif steer_input>MAX_STEER_DEGREES:
                        steer_input = MAX_STEER_DEGREES
                    sem_im = camera_data['sem_image']
                    img_counter += 1
                    time_grab = time.time_ns()
                    cv2.imwrite('out/semCam/%06d_%s_%s.png' % (time_grab, gen_dir_angle,round(steer_input,0)), sem_im)
                    image = camera_data['rgb_image']
                    image = cv2.putText(image, 'Steer: '+str(steer_input), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.imshow('RGB Camera',image)
                    if cv2.waitKey(0) == ord('q'):
                        quit = True
                        break
                    
        if quit:
            break
    exit_clean()