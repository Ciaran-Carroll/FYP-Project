from time import sleep
from picamera import PiCamera
from matplotlib import pyplot as plt
import numpy as np
import cv2

camera = PiCamera()
camera.resolution = (1024, 768)
camera.start_preview()

sleep(5) # gives 5 seconds to adjust camera before taking image

# Taking a normal image with the raspbery pi camera module
standardPhoto = camera.capture('/home/pi/Standard_Image/Normal_{:%Y_%m_%d_%H:%M}.jpg'.format(datetime.datetime.now()))
cv2.imwrite('/home/pi/Standard_Image/Normal_{:%Y_%m_%d_%H:%M}.jpg'.format(datetime.datetime.now()), standardPhoto)
