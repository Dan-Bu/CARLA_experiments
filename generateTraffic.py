import carla
from carla import command
import random
import glob
import os
import sys
import time
import logging
import carla.libcarla
from numpy import random

import settings

def generate_traffic(client, number_cars, number_pedestrians):

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    vehicles_list = []
    walkers_list = []
    all_id = []
    results = []

    client.set_timeout(10.0)
    settings.synchronous_master = True
    settings.asynch = False
    random.seed(0)

    world = client.get_world()

    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    world_settings = world.get_settings()
    traffic_manager.set_synchronous_mode(True)
    print("You are currently in asynchronous mode. If this is a traffic simulation, \
        you could experience some issues. If it's not working correctly, switch to synchronous \
        mode by using traffic_manager.set_synchronous_mode(True)")

    world.apply_settings(world_settings)

    blueprints = get_actor_blueprints(world, 'vehicle.*', 'All')
    if not blueprints:
        raise ValueError("Couldn't find any vehicles with the specified filters")
    blueprintsWalkers = get_actor_blueprints(world, 'walker.pedestrian.*', '2')
    if not blueprintsWalkers:
        raise ValueError("Couldn't find any walkers with the specified filters")

    
    blueprints = [x for x in blueprints if x.get_attribute('base_type') == 'car']

    blueprints = sorted(blueprints, key=lambda bp: bp.id)

    spawn_points = world.get_map().get_spawn_points()
    number_of_spawn_points = len(spawn_points)
    print(f"spawn points: {number_of_spawn_points}")
    if number_cars < number_of_spawn_points:
        random.shuffle(spawn_points)
    elif number_cars > number_of_spawn_points:
        msg = 'requested %d vehicles, but could only find %d spawn points'
        logging.warning(msg, number_cars, number_of_spawn_points)
        number_cars = number_of_spawn_points

    # @todo cannot import these directly.
    SpawnActor = command.SpawnActor
    SetAutopilot = command.SetAutopilot
    FutureActor = command.FutureActor

    # --------------
    # Spawn vehicles
    # --------------
    vehicle_batch = []
    hero = True
    for n, transform in enumerate(spawn_points):
        if n >= number_cars:
            break
        blueprint = random.choice(blueprints)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if hero:
            blueprint.set_attribute('role_name', 'hero')
            hero = False
        else:
            blueprint.set_attribute('role_name', 'autopilot')

        # spawn the cars and set their autopilot and light state all together
        vehicle_batch.append(SpawnActor(blueprint, transform))
        #vehicle_batch.append(SpawnActor(blueprint, transform)
        #    .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

    for response in client.apply_batch_sync(vehicle_batch, settings.synchronous_master):
        if response.error:
            logging.error(response.error)
        else:
            vehicles_list.append(response.actor_id)
    # Set automatic vehicle lights update if specified
    if False:
        all_vehicle_actors = world.get_actors(vehicles_list)
        for actor in all_vehicle_actors:
            traffic_manager.update_vehicle_lights(actor, True)

    # -------------
    # Spawn Walkers
    # -------------
    # some settings
    percentagePedestriansRunning = 0.7      # how many pedestrians will run
    percentagePedestriansCrossing = 0.9     # how many pedestrians will walk through the road
    #Option to have seeded pedestrian spawns and paths
    if False:
        world.set_pedestrians_seed(0)
        random.seed(0)
    # 1. take all the random locations to spawn
    spawn_points = []
    for i in range(1000):
        spawn_point = carla.Transform()
        loc = None
        while loc == None or loc in spawn_points:
            loc = world.get_random_location_from_navigation()
        if (loc != None) and not loc in spawn_points:
            spawn_point.location = loc 
            spawn_points.append(spawn_point)
    # 2. we spawn the walker object
    walker_batch = []
    walker_speed = []
    world = client.get_world()
    world_map = world.get_map()

    for pedestrian in range(0,number_pedestrians):
        walker_bp = random.choice(blueprintsWalkers)
        # set as not invincible
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        # set the max speed
        if walker_bp.has_attribute('speed'):
            if (random.random() > percentagePedestriansRunning):
                # walking
                speed = walker_bp.get_attribute('speed')
                walker_speed.append(str(float(walker_bp.get_attribute('speed').recommended_values[2]) + 16.0))
            else:
                # running
                walker_speed.append(str(float(walker_bp.get_attribute('speed').recommended_values[1]) + 8.0))
        else:
            print("Walker has no speed")
            walker_speed.append(0.0)
        walker = None
        try_count = 0
        crosswalk_spawnpoints = world_map.get_crosswalks()
        while walker == None and try_count < 20:
            spawn_point_n = carla.Transform()
            #spawn_point_n.rotation = carla.libcarla.Rotation(5.0,0.0,0.0)
            #spawn_point_n.location = world.get_random_location_from_navigation()
            spawn_point_n = random.choice(spawn_points)
            spawn_point_n.location.z += 1.0
            walker = world.try_spawn_actor(walker_bp, spawn_point_n)
            try_count += 1
        results.append(walker)
        #walker_batch.append(SpawnActor(walker_bp, spawn_point))
    #results = client.apply_batch_sync(walker_batch, True)
    walker_speed2 = []
    for i in range(len(results)):
        #if results[i].error:
        #    logging.error(results[i].error)
        #else:
            #walkers_list.append({"id": results[i].actor_id})
            walkers_list.append({"id": results[i].id})
            walker_speed2.append(walker_speed[i])
    walker_speed = walker_speed2
    # 3. we spawn the walker controller
    batch = []
    walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    for i in range(len(walkers_list)):
        batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
    results = client.apply_batch_sync(batch, True)
    for i in range(len(results)):
        if results[i].error:
            logging.error(results[i].error)
        else:
            walkers_list[i]["con"] = results[i].actor_id
    # 4. we put together the walkers and controllers id to get the objects from their id
    for i in range(len(walkers_list)):
        all_id.append(walkers_list[i]["con"])
        all_id.append(walkers_list[i]["id"])
    all_actors = world.get_actors(all_id)

    # wait for a tick to ensure client receives the last transform of the walkers we have just created
    if settings.asynch or not settings.synchronous_master:
        world.wait_for_tick()
    else:
        world.tick()
    

    # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
    # set how many pedestrians can cross the road
    world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
    for i in range(0, len(all_id), 2):
        # start walker
        actor = all_actors[i]
        all_actors[i].start()

        # set walk to random point that isn't our spawn
        location = None
        while all_actors[i].parent.bounding_box.location == location or location == None:
            #location = random.choice(crosswalk_spawnpoints)
            location = world.get_random_location_from_navigation()

        actor_controller_location = all_actors[i].parent.bounding_box.location
        print("Issues here?")
        all_actors[i].go_to_location(location) # <-- This is where the crash happens
        print("Nope")
        # max speed
        all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

        

    print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(vehicles_list), len(walkers_list)))

    # Example of how to use Traffic Manager parameters
    traffic_manager.global_percentage_speed_difference(30.0)
    return vehicles_list, walkers_list, all_id, batch


    


def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2, 3]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []
