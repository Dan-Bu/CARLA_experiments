import cv2

#Sets the mode - 0 is live data usage, 1 is recording mode to generate training data
generate_sem_seg_train_data = 1

#Training parameters
amount_training_data_to_generate = 50

#Camera 1 image size
image_h = 360
image_w = 640

#Simulator settings
synchronous_master = False
asynch = True

#Visualisation settings
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
text_color = (255, 255, 255) # White text
thickness = 1
org = (30, 30)  # Line one (Speed)
org2 = (30, 50) # Line two (Steering angle)
org3 = (30, 70) # Line three (Other telemetry)
