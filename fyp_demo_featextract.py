import numpy as np
import cv2
from matplotlib import pyplot as plt
import numpy.linalg as la
import math

# Reads the segmented image of the mole
img_seg1 = cv2.imread('img.jpg', cv2.IMREAD_UNCHANGED)

# Convert the BGR image to RGB
img_rgb1 = cv2.cvtColor(img_seg1, cv2.COLOR_BGR2RGB)

# Convert the image to grayscale
img_gray1 = cv2.cvtColor(img_seg1, cv2.COLOR_BGR2GRAY)

# Convert the image to HSV
img_hsv1 = cv2.cvtColor(img_seg1, cv2.COLOR_BGR2GRAY)

# Otsu thresholding
ret, thresh2 = cv2.threshold(img_gray1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(thresh2, kernel, iterations = 1)

edges = cv2.bitwise_xor(thresh2, erosion)

cv2.imshow('thresholded image', thresh2)
cv2.imshow('erosion', erosion)
cv2.imshow('edges', edges)

# Finding the contours in the image
im2, contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    moments = cv2.moments(cnt) # Calculate moments
    if moments['m00']!=0:
        # Approximation of the centre of the contour
        cx = int(moments['m10']/moments['m00'])   # cx = M10/M00
        cy = int(moments['m01']/moments['m00'])  # cy = M01/M00
        
        # Actual centre of the contour
        cx1 = (moments['m10']/moments['m00'])   # cx = M10/M00
        cy1 = (moments['m01']/moments['m00'])  # cy = M01/M00
        
        moment_area = moments['m00'] # Contour area from moment
        contour_area = cv2.contourArea(cnt) # Calculate using OpenCV's built in function
      
        area = moments['m00'] # Area of the output
        
        mu20 = moments['mu20']
        mu11 = moments['mu11']
        mu02 = moments['mu02']

        
        leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
        rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
        topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
        bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
        
        [lm_x, lm_y] = leftmost
        [rm_x, rm_y] = rightmost
        [tm_x, tm_y] = topmost
        [bm_x, bm_y] = bottommost
        
        cv2.line(im2, (cx, cy), (leftmost), (0,0,255), 1)
        cv2.line(im2, (cx, cy), (rightmost), (0,0,255), 1)
        cv2.line(im2, (cx, cy), (topmost), (0,0,255), 1)
        cv2.line(im2, (cx, cy), (bottommost), (0,0,255), 1)
    
        evals, evects = la.eigh(np.array([[mu20, mu11],[mu11, mu02]]))
        
        [eval1, eval2] = evals

        eccent = eval2/eval1
        
        w, l = 4*np.sqrt(evals/contour_area)
        
        if (l > w):
            diameter = l
        else:
            diameter = w

        # Orientation
        orient = 0.5*np.arctan((2*mu11)/(mu20-mu02))
        orientation = orient * 100
            
        cv2.drawContours(thresh2, [cnt],0,(0,0,0),1) # draws contours in green
        cv2.circle(im2,(cx,cy),1,(0,0,255),-1)
    
#    print("Leftmost point: ", leftmost)
#    print("Rightmost point: ", rightmost)
#    print("Topmost point: ", topmost)
#    print("Bottommost point: ", bottommost)
    cv2.circle(im2,(leftmost),1,(0,0,255),-1)
    cv2.circle(im2,(rightmost),1,(0,0,255),-1)
    cv2.circle(im2,(topmost),1,(0,0,255),-1)
    cv2.circle(im2,(bottommost),1,(0,0,255),-1)
    
    # Distance is the square root of (x2 - x1)^2 + (y2 - y1)^2
    distance_lm2centre = math.sqrt(((cx - lm_x)**2) + ((cy - lm_y)**2))
    distance_rm2centre = math.sqrt(((cx - rm_x)**2) + ((cy - rm_y)**2))
    distance_tm2centre = math.sqrt(((cx - tm_x)**2) + ((cy - tm_y)**2))
    distance_bm2centre = math.sqrt(((cx - bm_x)**2) + ((cy - bm_y)**2))
    
#    print("Distance from leftmost point to center: ", distance_lm2centre)
#    print("Distance from rightmost point to center: ", distance_rm2centre)
#    print("Distance from topmost point to center: ", distance_tm2centre)
#    print("Distance from bottommost point to center: ", distance_bm2centre)
    
    # Look thorugh again
        
    # Slope of points m = (y2 - y1) / (x2 - x1)
    # The order of the point won't change the results
    slope_lm2centre = ((lm_y - cy) / (lm_x - cx))
    slope_rm2centre = ((rm_y - cy) / (rm_x - cx))
    slope_tm2centre = ((tm_y - cy) / (tm_x - cx))
    slope_bm2centre = ((bm_y - cy) / (bm_x - cx))
                
    m_lm = slope_lm2centre * -1
    m_rm = slope_rm2centre * -1
    m_tm = slope_tm2centre * -1
    m_bm = slope_bm2centre * -1
        
#    print("Slope of leftmost point over center point: ", m_lm)
#    print("Slope of rightmost point over center point: ", m_rm)
#    print("Slope of topmost point over center point: ", m_tm)
#    print("Slope of bottommost point over center point: ", m_bm)
    
    # Be careful here - change 
    
    new_m_lm = m_lm * 1
    new_m_bm = m_bm * -1
    tetha_lmvbm = abs(math.degrees(math.atan((new_m_lm - new_m_bm)/(1 + (new_m_lm * new_m_bm)))))
    
    new_m_bm = m_bm * 1
    new_m_rm = m_rm * -1
    tetha_bmvrm = abs(math.degrees(math.atan((new_m_bm - new_m_rm)/(1 + (new_m_bm * new_m_rm)))))
    
    new_m_rm = m_rm * 1
    new_m_tm = m_tm * -1
    tetha_rmvtm = abs(math.degrees(math.atan((new_m_rm - new_m_tm)/(1 + (new_m_rm * new_m_tm)))))
    
    new_m_tm = m_tm * -1
    new_m_lm = m_lm * 1
    tetha_tmvlm = abs(math.degrees(math.atan((new_m_tm - new_m_lm)/(1 + (new_m_tm * new_m_lm)))))
    total = tetha_lmvbm + tetha_bmvrm + tetha_rmvtm + tetha_tmvlm
    tetha_accuracy = (total/360) * 100
    
#    print('Angle between the leftmost point and bottommost point: ', tetha_lmvbm,'in degrees')
#    print('Angle between the bottommost point and rightmost point: ', tetha_bmvrm,'in degrees')
#    print('Angle between the rightmost point and topmost point: ', tetha_rmvtm,'in degrees')
#    print('Angle between the topmost point and leftmost point: ', tetha_tmvlm,'in degrees')
#    print('Total amount of the angles accounted for', total,'in degrees')
#    print('Points were rounded to a integer number so a resionable error is acceptable')
#    print('Accuracy of angles',tetha_accuracy,'%')
        
    cv2.imshow('Output', im2)

print("Sample output for the program: ")
print("There is %d contour in the image which should represent our mole" % len(contours))

print("Moments of the contour: ")
print(moments)

print("Pixel area of the contour: ", area)

print("Diameter for the major axis of the contour: ", l)
print("Diameter for the minor axis of the contour: ", w)
print("Diameter of the contour: ", diameter)

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

print("Orientation of the contour: ", orientation,"degrees")

print("Eigenvalues: ")
print(evals)
print("Eigenvector: ")
print(evects)

print("Eccentricity of the contour: ", eccent)

print("Leftmost point: ", leftmost)
print("Rightmost point: ", rightmost)
print("Topmost point: ", topmost)
print("Bottommost point: ", bottommost)

print("Distance from leftmost point to center: ", distance_lm2centre)
print("Distance from rightmost point to center: ", distance_rm2centre)
print("Distance from topmost point to center: ", distance_tm2centre)
print("Distance from bottommost point to center: ", distance_bm2centre)    
    
print("Slope of leftmost point over center point: ", m_lm)
print("Slope of rightmost point over center point: ", m_rm)
print("Slope of topmost point over center point: ", m_tm)
print("Slope of bottommost point over center point: ", m_bm)
        
print('Angle between the leftmost point and bottommost point: ', tetha_lmvbm,'in degrees')
print('Angle between the bottommost point and rightmost point: ', tetha_bmvrm,'in degrees')
print('Angle between the rightmost point and topmost point: ', tetha_rmvtm,'in degrees')
print('Angle between the topmost point and leftmost point: ', tetha_tmvlm,'in degrees')
print('Total amount of the angles accounted for', total,'in degrees')
print('Points were rounded to a integer number so a resionable error is acceptable')
print('Accuracy of angles',tetha_accuracy,'%')
    
cv2.waitKey(0)
cv2.destroyAllWindows()