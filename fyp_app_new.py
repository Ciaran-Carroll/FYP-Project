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

## Screenshot 1

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

# Taking a normal image of the mole with the raspbery pi camera module
standardPhoto = 
camera.capture('/home/pi/Standard_Image/Normal_{:%Y_%m_%d_%H:%M}.jpg'
.format(datetime.datetime.now()))

cv2.imwrite('/home/pi/Standard_Image/Normal_{:%Y_%m_%d_%H:%M}.jpg'
.format(datetime.datetime.now()))

## Screenshot 2

# Convert the captured image to grayscale
imgGray = cv2.cvtColor(standardPhoto, cv2.COLOR_BGR2GRAY)

# Perform Otsu Thresholding on the grayscale image
ret, thresh = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY_INV
+cv2.THRESH_OTSU)

# Use Morphological opening to remove noise and reduce the effect of hairs
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 1)

# Finding the sure background area
sure_background = cv2.dialate(opening, kernel, iterations = 1)

# Finding the sure foreground area
distance_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
ret, sure_foreground = cv2.threshold(distance_transform, 
0.7*distance_transform.max(), 255, 0)

# Finding the unknown region
sure_foreground = np.uint8(sure_foreground)
unknown_region = cv2.subtract(sure_background, sure_foreground)

## Screenshot 3

# Marker labelling
retval, markers = cv2.connectedComponents(sure_foreground)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the unknown region zero
markers[unknown_region==255] = 0

# Apply the watershed and convert the resultto a uint8 image
markers = cv2.watershed(standardPhoto, markers)

# Setting the unwanted regions to appear black
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

# Displays and saves the segmented image
cv2.imshow('Watershed', res)
cv2.imwrite('/home/pi/Image_Watershed/Watershed_{:%Y_%m_%d_%H:%M}.jpg'
.format(datetime.datetime.now()),standardPhoto)

# Displays the regions detected by watershed algorithm
imgplt = plt.imshow(markers)
plt.colorbar()
plt.show()

## Screenshot 4

# Reads the segmented image of the mole
img_seg1 = img

# Convert the segemted image to grayscale
img_gray1 = cv2.cvtColor(img_seg1, cv2.COLOR_BGR2GRAY)

# Use Otsu thresholding to convert the segmented image to a binary image
ret, thresh2 = cv2.threshold(img_gray1, 0, 255, cv2.THRESH_BINARY+
cv2.THRESH_OTSU)

# Shows the binary image
cv2.imshow('thresholded image', thresh2)

# Feature Extraction of the segmented image

# Finding the contours in the binary image
# The image should contain only one contours
im2, contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_TREE, 
cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    moments = cv2.moments(cnt) # Calculates image moments
    if moments['m00']!=0:
        # Approximation of the centre of the contour
        cx = int(moments['m10']/moments['m00'])   # cx = M10/M00
        cy = int(moments['m01']/moments['m00'])  # cy = M01/M00
        
        # Actual centre of the contour
        cx1 = (moments['m10']/moments['m00'])   # cx = M10/M00
        cy1 = (moments['m01']/moments['m00'])  # cy = M01/M00
        
        # Calculates the area of the contour
        area = moments['m00']
        round_area = round(area, 2)
        
        # Finds second order central moments
        mu20 = moments['mu20']
        mu11 = moments['mu11']
        mu02 = moments['mu02']
        
        # Calculate the covariance matrix of the contour
        covar_mat = np.array([[mu20, mu11],[mu11, mu02]])
        
        # Calculates the eigenvalues and eigenvectors of the covariance matrix
        evals, evects = la.eigh(covar_mat)
        
        # Seperates the eigenvalues in the matrix
        [eval1, eval2] = evals
        
        # Calculates the eccentricity of the contour
        eccent = eval1/eval2
        
        # Finds the major and minor axis of the contour
        w, l = 4*np.sqrt(evals/area)
        
        # Sets the diameter of the contour to the longer axis
        if (l > w):
            diameter = l
        else:
            diameter = w
        
        # Calculates the orientation of the contour
        orient = 0.5*np.arctan((2*mu11)/(mu20-mu02))
        orientation = orient * 100
        
		## Screenshot 5
		
        # Finds the four most extreme points in the contour
        leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
        rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
        topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
        bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
		
        # Draw a circle at each of the extreme points in black
        cv2.circle(im2,(leftmost),1,(0,0,0),-1)
        cv2.circle(im2,(rightmost),1,(0,0,0),-1)
        cv2.circle(im2,(topmost),1,(0,0,0),-1)
        cv2.circle(im2,(bottommost),1,(0,0,0),-1)
        
        # Seperates the x and y points in each of the extreme points
        [lm_x, lm_y] = leftmost
        [rm_x, rm_y] = rightmost
        [tm_x, tm_y] = topmost
        [bm_x, bm_y] = bottommost
        
        # Draws a line from each extreme point to the centre in black
        cv2.line(im2, (cx, cy), (leftmost), (0,0,0), 1)
        cv2.line(im2, (cx, cy), (rightmost), (0,0,0), 1)
        cv2.line(im2, (cx, cy), (topmost), (0,0,0), 1)
        cv2.line(im2, (cx, cy), (bottommost), (0,0,0), 1)
        
        # Calculates the distance from each extreme point to the center
        # Using the distance formula = square root of ((x2 - x1)^2 + (y2 - y1)^2)
        distance_lm2centre = math.sqrt(((cx - lm_x)**2) + ((cy - lm_y)**2))
        distance_rm2centre = math.sqrt(((cx - rm_x)**2) + ((cy - rm_y)**2))
        distance_tm2centre = math.sqrt(((cx - tm_x)**2) + ((cy - tm_y)**2))
        distance_bm2centre = math.sqrt(((cx - bm_x)**2) + ((cy - bm_y)**2))
        
        # Calculate the slope of each line from extreme point to the center
        # Slope of points: m = (y2 - y1) / (x2 - x1)
        # The order of the points doesn't affect our results
        slope_lm2centre = ((lm_y - cy) / (lm_x - cx))
        slope_rm2centre = ((rm_y - cy) / (rm_x - cx))
        slope_tm2centre = ((tm_y - cy) / (tm_x - cx))
        slope_bm2centre = ((bm_y - cy) / (bm_x - cx))
        
        # Since the images is viewed in the opposite direction
        # Each of the slopes calculates are the inverse of the expected values
        # Multiplying each slope by -1 solves this problem
        m_lm = slope_lm2centre * -1
        m_rm = slope_rm2centre * -1
        m_tm = slope_tm2centre * -1
        m_bm = slope_bm2centre * -1
        
        # Calculates the angles between each of the extreme points
        # Setting the slopes in order to view the image in normal coordinates
        new_m_lm = m_lm * 1
        new_m_bm = m_bm * -1
        tetha_lmvbm = abs(math.degrees(math.atan((new_m_lm - new_m_bm)
		/(1 + (new_m_lm * new_m_bm)))))
        
        new_m_bm = m_bm * 1
        new_m_rm = m_rm * -1
        tetha_bmvrm = abs(math.degrees(math.atan((new_m_bm - new_m_rm)
		/(1 + (new_m_bm * new_m_rm)))))
        
        new_m_rm = m_rm * 1
        new_m_tm = m_tm * -1
        tetha_rmvtm = abs(math.degrees(math.atan((new_m_rm - new_m_tm)
		/(1 + (new_m_rm * new_m_tm)))))
        
        new_m_tm = m_tm * -1
        new_m_lm = m_lm * 1
        tetha_tmvlm = abs(math.degrees(math.atan((new_m_tm - new_m_lm)
		/(1 + (new_m_tm * new_m_lm)))))
        
		# Calculates the total angle between each of the extreme points 
		# and accuracy of results
        total = tetha_lmvbm + tetha_bmvrm + tetha_rmvtm + tetha_tmvlm
        tetha_accuracy = (total/360) * 100
        
		# Draws the contour and centre point in black
        cv2.drawContours(thresh2, [cnt],0,(0,0,0),1)
        cv2.circle(im2,(cx,cy),1,(0,0,0),-1)
        
    cv2.imshow('Output', im2)

## Done
	
print("Sample output for the program: ")
print("There is %d contour in the image which should represent our mole" % len(contours))

print("Moments of the contour: ")
print(moments)

print("Pixel area of the contour: {0:.2f}".format(area))

print("Diameter for the major axis of the contour: ", l)
print("Diameter for the minor axis of the contour: ", w)
print("Diameter of the contour: {0:.2f}".format(diameter))

print("Actual Central moments of the contour are:")
print("    cx = ", cx1)
print("    cy = ", cy1)

print("Approximate Central moments of the contour are:")
print("    cx = ", cx)
print("    cy = ", cy)

print("Center point of the contour: ", (cx, cy))

print("Second central moments of the contour are:")
print("    mu20 = ", moments['m20'])
print("    mu11 = ", moments['m11'])
print("    mu02 = ", moments['m02'])

print("Covariance matrix of the contour: ")
print(covar_mat)

print("Orientation of the contour: {0:.2f}".format(orientation),"degrees")

print("Eigenvalues: ")
print(evals)
print("Eigenvector: ")
print(evects)

print("Eccentricity of the contour: {0:.2f}".format(eccent))

print("Leftmost point: ", leftmost)
print("Rightmost point: ", rightmost)
print("Topmost point: ", topmost)
print("Bottommost point: ", bottommost)

print("Distance from leftmost point to center: {0:.2f}".format(distance_lm2centre))
print("Distance from rightmost point to center: {0:.2f}".format(distance_rm2centre))
print("Distance from topmost point to center: {0:.2f}".format(distance_tm2centre))
print("Distance from bottommost point to center: {0:.2f}".format(distance_bm2centre))    

print("Slope of leftmost point over center point: {0:.2f}".format(m_lm))
print("Slope of rightmost point over center point: {0:.2f}".format(m_rm))
print("Slope of topmost point over center point: {0:.2f}".format(m_tm))
print("Slope of bottommost point over center point: {0:.2f}".format(m_bm))

print("Angle between the leftmost point and bottommost point: {0:.2f}".format(tetha_lmvbm),"in degrees")
print("Angle between the bottommost point and rightmost point: {0:.2f}".format(tetha_bmvrm),"in degrees")
print("Angle between the rightmost point and topmost point: {0:.2f}".format(tetha_rmvtm),"in degrees")
print("Angle between the topmost point and leftmost point: {0:.2f}".format(tetha_tmvlm),"in degrees")
print("Total amount of the angles accounted for {0:.2f}".format(total),"in degrees")
print("Points were rounded to a integer number so a resionable error is acceptable")
print("Accuracy of angles {0:.2f}".format(tetha_accuracy),"%")

cv2.waitKey(0)
cv2.destroyAllWindows()
