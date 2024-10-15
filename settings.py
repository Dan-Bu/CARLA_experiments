import cv2

#Sets the mode 
# - "CameraSegmentationTestrun" is live camera data segmentation
# - "GenerateData" is recording mode to generate training data
# - "test_single_image" runs a single image test on the segmentation network
# - "TrainCameraDriving" trains the self-driving operator
# - "CameraDriving" runs the self-driving operator
mode = "CameraSegmentationTestrun"
images_from_fixed_routes = False # This also means no traffic other than our car.

#Training set parameters
amount_training_data_to_generate = 1500
sem_seg_data_path = './out/semseg'
epochs = 64
dataloader_num_workers = 8

#Camera 1 image size
image_h = 400
image_w = 640

#Simulator settings
synchronous_master = False
asynch = True

#Visualisation settings
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
text_color = (255, 255, 255) # White text
thickness = 1
text_loc1 = (30, 30)  # Line one (Speed)
text_loc2 = (30, 50) # Line two (Steering angle)
text_loc3 = (30, 70) # Line three (Other telemetry)

color_to_class = {
            (0, 0, 0): 0,       # Unlabeled
            (128, 64, 128): 1,  # Road
            (244, 35, 232): 2,  # Sidewalk
            (70, 70, 70): 3,    # Building
            (102, 102, 156): 4, # Wall
            (190, 153, 153): 5, # Fence
            (153, 153, 153): 6, # Pole
            (250, 170, 30): 7,  # Traffic light
            (220, 220, 0): 8,   # Traffic sign
            (107, 142, 35): 9,  # Vegetation
            (152, 251, 152): 10,# Terrain
            (70, 130, 180): 11, # Sky
            (220, 20, 60): 12,  # Pedestrian
            (255, 0, 0): 13,    # Rider
            (0, 0, 142): 14,    # Car
            (0, 0, 70): 15,     # Truck
            (0, 60, 100): 16,   # Bus
            (0, 60, 100): 17,   # Train
            (0, 0, 230): 18,    # Motorcycle
            (119, 11, 32): 19,  # Bicycle
            (110, 190, 160): 20,# Static
            (170, 120, 50): 21, # Dynamic
            (55, 90, 80): 22,   # Other
            (45, 60, 150): 23,  # Water
            (157, 234, 50): 24, # Road line
            (81, 0, 81): 25,    # Ground
            (150, 100, 100): 26,# Bridge
            (230, 150, 140): 27,# Rail Track 
            (180, 165, 180): 28 # Guard Rail     
            # Add more color-class mappings as needed
        }
