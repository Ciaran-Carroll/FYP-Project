## -----------------------------------------------------------------------------------
##
## Student name:   	Ciaran Carroll
## Student Id:  	13113259
##
## Implementation of Image Segmentation using  Morphological Watershed
##
## Uses a test image to develop a working prototype
##
##--------------------------------------------------------------------------------------

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Loads the test image
img = cv2.imread('Mole_testimage.png') # Change location to raspberry pi location
#cv2.imshow('Original Image', img)

# Convert the image to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Image Segmentation using the watershed algorithm
# Apply Otsu thresholding on the grayscale image
retval, threshold = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Use Morphological opening to remove noise and reduce the effect of hairs
# Is also prevents oversegmentation by removing small holes
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN,kernel, iterations = 1)

# Use Morphological dialation to find areas of the image that are surely background
sure_background = cv2.dilate(opening, kernel, iterations=1)

# Use Euclidean distance transform to calculates the distance from every binary image
# to the nearest zero pixel
distance_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)

# Apply a threshold to the distance transform to determine with a high probability
retval, sure_foreground = cv2.threshold(distance_transform,0.7*distance_transform.max(),255,0)

# Convert the sure foreground image from a float32 array to uint8
# Finding the unknown region by subtracting the sure forground from sure background
sure_foreground = np.uint8(sure_foreground)
unknown = cv2.subtract(sure_background,sure_foreground)

# Marker labelling to build the dams (barriers) to stop the water from merging
# Labelling the background region 0, and the other objects with integers starting with 1
retval, markers = cv2.connectedComponents(sure_foreground)

# Add one to all labels so that sure background is not 0, but 1
# This is so that the watershed doesn't consider it as the unknown region
markers = markers+1

# Label the unknown region zero
markers[unknown==255] = 0

# Now we let the water fall and our barriers be drawn in red
# Apply the watershed and convert the result back into uint8 image
markers = cv2.watershed(img, markers)

# The watershed algorithm labels each region order of size
# The second region should be our mole
img[markers==-1] = (0,0,255) # Boundary region
#cv2.imshow('Showing label markers on Original image', img)
img[markers==1] = (0,0,0) # Background region
#cv2.imshow('Background removed from Original image', img)
img[markers==-1] = (0,0,0) # Boundary region
img[markers==3] = (0,0,0)
img[markers==4] = (0,0,0)
img[markers==5] = (0,0,0)
img[markers==6] = (0,0,0)
img[markers==7] = (0,0,0)
img[markers==8] = (0,0,0)
img[markers==9] = (0,0,0)
img[markers==10] = (0,0,0)

# Displays and saves Segmented image
cv2.imshow('Segmented Image', img)
cv2.imwrite('Segmented_mole.jpg', img)

# Displays the regions detected by watershed
imgplt = plt.imshow(markers)
plt.colorbar()
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
