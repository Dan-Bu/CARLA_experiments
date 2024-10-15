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
import cameraImageSegmentation

#sys.path.append('C:\\Users\\Daniel\\Documents\\PrivateProjects\\CARLA\\CARLA_0.9.15\\WindowsNoEditor\\PythonAPI\\carla') # tweak to where you put carla
sys.path.append('C:/Users/Daniel/Documents/PrivateProjects/CARLA/CARLA_0.9.15/WindowsNoEditor/PythonAPI/carla') # tweak to where you put carla
from agents.navigation.global_route_planner import GlobalRoutePlanner



if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium') # To utilize the 4070 TI Supers tensor cores

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

        exit()

    if settings.mode == "CameraDriving":

        exit()




