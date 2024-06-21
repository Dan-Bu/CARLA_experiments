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

from torchvision import transforms
from torch.utils.data import DataLoader
import datetime
import re

import settings
import generateTraffic
import generateTrainingImages
import dataloader.rgbcameradataloader as customDataset
import networks.camera.base_model as base_model

#sys.path.append('C:\\Users\\Daniel\\Documents\\PrivateProjects\\CARLA\\CARLA_0.9.15\\WindowsNoEditor\\PythonAPI\\carla') # tweak to where you put carla
sys.path.append('C:/Users/Daniel/Documents/PrivateProjects/CARLA/CARLA_0.9.15/WindowsNoEditor/PythonAPI/carla') # tweak to where you put carla
from agents.navigation.global_route_planner import GlobalRoutePlanner

# Camera Callback. Defines what the camera should do when a frame is generated.
def camera_callback(image, data_dictionary):
    #Define the offsets to cut the image
    data_dictionary['image'] = np.reshape(np.copy(image.raw_data),(image.height,image.width,4))

# Callback for semantic segmentation camera
def sem_callback(image,data_dictionary):
    ########## IMPORTANT CHANGE for Semantic camera ##############
    image.convert(carla.ColorConverter.CityScapesPalette)
    data_dictionary['sem_image'] = np.reshape(np.copy(image.raw_data),(image.height,image.width,4))

#Training function for the unet
def train_UNet():
    # Train LightningModule
    max_epochs = settings.epochs

    data_module = customDataset.CustomDataLoader(root_dir=settings.sem_seg_data_path, batch_size=8)
    model = base_model.LightningUNet(in_channels=3, out_channels=29)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss')
    tb_logger = pl_loggers.TensorBoardLogger("lightning_logs", name="tb_log", default_hp_metric=False)
    trainer = pl.Trainer(accelerator="gpu", devices=[0], max_epochs=max_epochs, callbacks=[checkpoint_callback], logger=tb_logger)
    trainer.fit(model, data_module)
    train_acc = trainer.callback_metrics['train_accuracy'].item()
    val_acc = trainer.callback_metrics['val_accuracy'].item()

    # Save the trained model
    date = datetime.datetime.now().strftime('%d-%m-%Y')
    newpath = f'./checkpoints/unet/{date}_{max_epochs}Epochs' 
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    highest_number = 0
    for filename in os.listdir(newpath):
        if filename.endswith('.pth'):
            try:
                number = int(re.findall("\d+", filename.split('.')[0])[0])
                if number > highest_number:
                    highest_number = number
            except ValueError:
                continue
                  
    # Test the model
    trainer.test(model, data_module)
    test_acc = trainer.callback_metrics['test_accuracy'].item()
    # Print the final accuracy
    print(f"Final Training Accuracy: {train_acc}")
    print(f"Final Validation Accuracy: {val_acc}")
    print(f"Final Test Accuracy: {test_acc}")

def test_on_single_image():
    # define color mask
    reverse_dictionary = {v:k for k,v in settings.color_to_class.items()}


    # display the camera and segmentation image
    data_module = customDataset.CustomDataLoader(root_dir=settings.sem_seg_data_path, batch_size=8)
    data_module.setup()
    image = data_module.train_dataset.__getitem__(0)
    to_pil = transforms.ToPILImage()
    visual_img = to_pil(image[0])
    visual_mask = image[1].argmax(dim=1)
    class_indices_mask = np.zeros((settings.image_h, settings.image_w, 3), dtype=np.uint8)
    for class_idx, color in reverse_dictionary.items():
        class_indices_mask[visual_mask[0] == class_idx] = color
    visual_mask = class_indices_mask
    visual_mask = to_pil(visual_mask)
    
    
    # run the network
    base_model.LightningUNet(in_channels=3, out_channels=29)
    model = base_model.LightningUNet.load_from_checkpoint("lightning_logs/tb_log/version_5/checkpoints/epoch=3-step=3152.ckpt", in_channels=3, out_channels=29)   
    model.eval()
    with torch.no_grad():
        input_image = image[0].unsqueeze(0)
        input_image = input_image.cuda()
        output_tensor = model(input_image)
    # Map the colors to class indices
    class_indices = np.zeros((settings.image_h, settings.image_w, 3), dtype=np.uint8)
    output_tensor = output_tensor.cpu().argmax(dim=1)
    for class_idx, color in reverse_dictionary.items():
        class_indices[output_tensor[0] == class_idx] = color
    class_indices = to_pil(class_indices)
    visual_img.show()
    visual_mask.show()
    class_indices.show()


def testrun():
    # Setup our data dictionary
    data_dict = {
                'image' : np.zeros((settings.image_w, settings.image_h)),
                'sem_image' : np.zeros((settings.image_w, settings.image_h)),  
                'image_w' : settings.image_w, 
                'image_h': settings.image_h
                }
    
    if settings.mode == "TrainSemUNet":
        train_UNet()
        exit()
    # Connect to the client and retrieve the world object
    client = carla.Client('localhost', 2000)
    client.load_world("Town02", reset_settings=False, map_layers=carla.MapLayer.All)
    world = client.get_world()
    grp = GlobalRoutePlanner(world.get_map(), 10)

    actor_list = world.get_actors()
    for actor in actor_list:
        if "walker" in actor.type_id or "vehicle" in actor.type_id:
            actor.destroy()


    #-----------spawn NPC vehicles-----------
    # Get the blueprint library and filter for the vehicle blueprints
    vehicle_blueprints = world.get_blueprint_library().filter('*vehicle*')

    # Get the map's spawn points
    spawn_points = world.get_map().get_spawn_points()

    # Spawn 50 vehicles randomly distributed throughout the map 
    # for each spawn point, we choose a random vehicle from the blueprint library
    # Return values are lists of the vehicles, walkers, and all entities
    vehicles_list, walkers_list, all_id, batch = generateTraffic.generate_traffic(client, 50, 50)

    #-----------Spawn Agent-----------    
    #Spawn the "Ego Vehicle", aka the one that we occupy as our agent.
    ego_bp = world.get_blueprint_library().find('vehicle.dodge.charger_2020')

    ego_bp.set_attribute('role_name', 'hero')
    ego_vehicle = None
    while ego_vehicle == None:
        ego_vehicle = world.try_spawn_actor(ego_bp, random.choice(spawn_points))

    #camera mount offset on the car - you can tweak these to have the car in view or not
    CAMERA_POS_Z = 1.5
    CAMERA_POS_X = 0.5
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '640') # this ratio works in CARLA 9.14 on Windows
    camera_bp.set_attribute('image_size_y', '400')
    camera_init_trans = carla.Transform(carla.Location(z=CAMERA_POS_Z,x=CAMERA_POS_X))
    #this creates the camera in the sim
    camera = world.spawn_actor(camera_bp,camera_init_trans,attach_to=ego_vehicle)

    # Start camera with PyGame callback
    if False:
        camera.listen(lambda image: image.save_to_disk('out/camera1/%06d.png' % image.frame))
    else:
        image_w = camera_bp.get_attribute('image_size_x').as_int()
        image_h = camera_bp.get_attribute('image_size_y').as_int()
        
        # this actually opens a live stream from the camera
        camera.listen(lambda image: camera_callback(image, data_dict))

    # setting semantic camera
    camera_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
    camera_bp.set_attribute('image_size_x', '640') # this ratio works in CARLA 9.13 on Windows
    camera_bp.set_attribute('image_size_y', '400')
    camera_bp.set_attribute('fov', '90')
    camera_init_trans = carla.Transform(carla.Location(z=CAMERA_POS_Z,x=CAMERA_POS_X))
    camera_sem = world.spawn_actor(camera_bp,camera_init_trans,attach_to=ego_vehicle)
    image_w = 640
    image_h = 400


    # this actually opens a live stream from the sem_camera
    camera_sem.listen(lambda image: sem_callback(image,data_dict))


    # Set autopilot on for all vehicles so they drive around.
    tm = client.get_trafficmanager(8000)
    tm_port = tm.get_port()
    for vehicle in world.get_actors().filter('*vehicle*'):
        vehicle.set_autopilot(True, tm_port)


    #Setup a CV2 window
    cv2.namedWindow('RGB Camera', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RGB Camera', data_dict['image'])

    cv2.namedWindow('Live Segmentation', cv2.WINDOW_AUTOSIZE)
    quit = False
    show_sem_seg = False

    base_model.LightningUNet(in_channels=3, out_channels=29)
    model = base_model.LightningUNet.load_from_checkpoint("lightning_logs/tb_log/version_6/checkpoints/epoch=55-step=44128.ckpt", in_channels=3, out_channels=29)   
    model.eval()
    # define color mask
    to_pil = transforms.ToPILImage()
    while True:
            #---Advance the world tick---
            world.tick()

            #---Check for the exit command (User pressing q)---
            if cv2.waitKey(10)==ord('q'):
                quit = True
                break
            elif cv2.waitKey(10)==ord('t'):
                if show_sem_seg == False:
                    show_sem_seg = True
                else: 
                    show_sem_seg = False
        
            #---Vehicle control---
            v = ego_vehicle.get_velocity()
            speed = round(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2), 1)

            #---Visualization of onboard camera---
            if show_sem_seg == False:
                image = data_dict['image']
            else: 
                image = data_dict['sem_image']



            # preprocess the image from the sensor.
            image_rgb = torch.from_numpy(data_dict['image'])
            image_rgb = image_rgb[:, :, 0:3]            # Remove alpha channel
            image_rgb = image_rgb[:,:,[2,1,0]]          # Bring image in RGB from BGR format because CV2 is a special bee
            image_rgb = image_rgb / 255.0               # Normalize to [0,1]
            image_rgb = image_rgb.permute(2, 0, 1)      # Permute to C,H,W
            image_rgb = image_rgb.unsqueeze(0).cuda()   # Add batch dimension


            
            #evaluate network
            with torch.no_grad():
                output_tensor = model(image_rgb).cpu().argmax(dim=1).cpu()

            # Map Class to color
            segmentation_image = map_colors(output_tensor)
            #add speed display
            image = cv2.putText(image, f'Speed: {str(int(speed))} kmh', settings.text_loc1, settings.font, settings.font_scale, settings.text_color, settings.thickness, cv2.LINE_AA)
            #Render frame
            cv2.imshow('RGB Camera', image)
            segmentation_image = segmentation_image[:,:,[2,1,0]] # CV2 being a special BGR bee again.
            cv2.imshow('Live Segmentation', segmentation_image)

            

    if not settings.asynch and settings.synchronous_master:
        world_settings = world.get_settings()
        world_settings.synchronous_mode = False
        world_settings.no_rendering_mode = False
        world_settings.fixed_delta_seconds = None
        world.apply_settings(world_settings)

    print('\ndestroying %d vehicles' % len(vehicles_list))
    client.apply_batch([command.DestroyActor(x) for x in vehicles_list])
    print('\ndestroying %d walkers' % len(walkers_list))
    client.apply_batch([command.DestroyActor(x) for x in walkers_list])
    # stop walker controllers (list is [controller, actor, controller, actor ...])
    all_actors = world.get_actors(all_id)
    for i in range(0, len(all_id), 2):
        all_actors[i].stop()

    print('\ndestroying %d walkers' % len(walkers_list))
    client.apply_batch([command.DestroyActor(x) for x in all_id])

    time.sleep(0.5)


def map_colors(seg_output):
    reverse_dictionary = {v:k for k,v in settings.color_to_class.items()}
    # Map the colors to class indices
    class_indices = np.zeros((settings.image_h, settings.image_w, 3), dtype=np.uint8)
    for class_idx, color in reverse_dictionary.items():
        class_indices[seg_output[0] == class_idx] = color

    return class_indices

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium') # To utilize the 4070 TI Supers tensor cores

    if settings.mode == "test_single_image":
        test_on_single_image()
        exit()


    if settings.mode == "GenerateData":
        generateTrainingImages.generate_training_data()
        exit()

    if settings.mode == "testrun":
        testrun()
        exit()




