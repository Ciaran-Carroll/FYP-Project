# Student Name: 	Ciaran Carroll
# Student Id: 		13113259
# 
# This program has two purposes:
# (1) Seperates red colour values from the image. This feature could indicate that
#     the mole is bleeding or identify potentially dangerous colour features in the image.
# (2) Plot a histogram which represents the distribution of pixels intensity in the coloured image 
#     This would be useful to see the colour variation in the image which often occurs with Melanoma.
# 
# In our prototype, it is diffucult to test for this as all the moles available for testing look mostly normal.

import numpy as np
import cv2
from matplotlib import pyplot as plt

# Reads image
#img = cv2.imread('download.png', cv2.IMREAD_UNCHANGED)
img = cv2.imread('Red-Mole.jpg', cv2.IMREAD_UNCHANGED)
#img = cv2.imread('Melanoma.jpg', cv2.IMREAD_UNCHANGED)

# Convert the image to HSV
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Lower mask (0-10)
lower_red = np.array([0,50,50])
upper_red = np.array([10,255,255])
mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

# Upper mask (170-180)
lower_red = np.array([170,50,50])
upper_red = np.array([180,255,255])
mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

# Combine both masks
mask = mask0+mask1

# Set the output img to zero everywhere except the mask
output_img = img.copy()
output_img[np.where(mask==0)] = 0

## or your HSV image, which I believe is what you want
#output_hsv = img_hsv.copy()
#output_hsv[np.where(mask==0)] = 0

cv2.imshow('Original image', img)
#cv2.imshow('HSV image', img_hsv)
cv2.imshow('Only allowing red colours in the image', output_img)
#cv2.imshow('Only red colours passed through the image1', output_hsv)

# Plots a histogram of the segmented image
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    hist = cv2.calcHist([img], [0], None, [256], [0,256])
    plt.plot(hist, color = col)
    plt.xlim([0, 256])
plt.title('Histogram for the segmented image')
plt.show()

cv2.waitKey(0)
cvv2.destroyAllWindows()