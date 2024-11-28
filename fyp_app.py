## ---------------------------------------------------------------------------
##
## Date: 			28-01-2018
##
## Student Name: 	   	Ciaran Carroll
## Student Id: 			13113259
## Language: 		   	Python
## Libraries used: 		Numpy, Matplotlib, OpenCV
##
## Project title: 	Melanoma (Skin cancer) Long term monitoring
##
## Operating System - Raspbian
##
## Project Description:
##
## Captures video from the Raspberry Pi camera module
## Sleeps for 5 seconds giving time to adjust the camera
## Takes an image with the Raspberry Pi camera module
## Converts the image from RGB to HSV
##
##
## ---------------------------------------------------------------------------

from time import sleep
from picamera import PiCamera
from matplotlib import pyplot as plt
import numpy as np
import cv2
import numpy.linalg as la
import math

camera = PiCamera()
camera.resolution = (1024, 768)
camera.start_preview()

sleep(5) # gives 5 seconds to adjust camera before taking image

# Taking a normal image with the raspbery pi camera module
standardPhoto = camera.capture('/home/pi/Standard_Image/Normal_{:%Y_%m_%d_%H:%M}.jpg'.format(datetime.datetime.now()))
cv2.imwrite('/home/pi/Standard_Image/Normal_{:%Y_%m_%d_%H:%M}.jpg'.format(datetime.datetime.now()))

# Convert the captured image to grayscale
imgGray = cv2.cvtColor(standardPhoto, cv2.COLOR_BGR2GRAY)

# Perform Otsu Thresholding on the thresholded image
ret, thresh = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Use Morphological opening to remove noise and reduce the effect of hair on the image
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 1)

# Finding the sure background area
sure_background = cv2.dialate(opening, kernel, iterations = 1)

# Finding the sure foreground area
distance_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
ret, sure_foreground = cv2.threshold(distance_transform, 0.7*distance_transform.max(), 255, 0)

# Finding the unknown region
sure_foreground = np.uint8(sure_foreground)
unknown_region = cv2.subtract(sure_background, sure_foreground)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_foreground)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the unknown region zero
markers[unknown_region==255] = 0

# Apply the watershed and convert the resultto a uint8 image
markers = cv2.watershed(standardPhoto, markers)
m = cv2.convertScaleAbs(markers)

# Setting the unwanted regions to black
# Marker 2 should be the mole
standardPhoto[markers==-1]=(0,0,0)
standardPhoto[markers==1]=(0,0,0) # background region
standardPhoto[markers==3]=(0,0,0)
standardPhoto[markers==4]=(0,0,0)
standardPhoto[markers==5]=(0,0,0)
standardPhoto[markers==6]=(0,0,0)
standardPhoto[markers==7]=(0,0,0)
standardPhoto[markers==8]=(0,0,0)
standardPhoto[markers==9]=(0,0,0)
standardPhoto[markers==10]=(0,0,0)

# Threshold the image to get the mask and perform a bitwise_with the input image
ret, thresh2 = cv2.threshold(m,0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
res = cv2.bitwise_and(img,img, mask = thresh2)

cv2.imshow('Watershed', res)
cv2.imwrite('/home/pi/Image_Watershed/Watershed_{:%Y_%m_%d_%H:%M}.jpg'.format(datetime.datetime.now()),standardPhoto)
cv2.imwrite('/home/pi/Segmented_Image/Segmented_{:%Y_%m_%d_%H:%M}.jpg'.format(datetime.datetime.now()),standardPhoto)
imgplt = plt.imshow(markers)
plt.colorbar()
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
