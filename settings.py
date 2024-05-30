import cv2

#Sets the mode - "Live" is live data usage, "GenerateData" is recording mode to generate training data, "TrainSemUNet" trains the unet for semseg
mode = "TrainSemUNet"
images_from_fixed_routes = False # This also means no traffic other than our car.

#Training set parameters
amount_training_data_to_generate = 1500
sem_seg_data_path = './out/semseg'
epochs = 42
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
