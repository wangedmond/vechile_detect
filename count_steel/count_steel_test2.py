# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:26:29 2019

@author: wange
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 17:48:31 2019

@author: wange
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 22:59:02 2019

@author: wange
"""

import cv2
import numpy as np

filenama='D:/count_steel/test1.png'
#filenama='D:/test.jpg'
img=cv2.imread(filenama,0)
print(img)
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 2)
ret,erosion=cv2.threshold(erosion,105,255,cv2.THRESH_BINARY)  
#blur = cv2.GaussianBlur(erosion,(5,5),0)
#ret,erosion = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#ret,erosion = cv2.threshold(erosion,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
erosion = cv2.erode(erosion,kernel,iterations =2 )
#erosion = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
#erosion = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
#erosion = cv2.dilate(erosion,kernel,iterations = 1)
#erosion = cv2.erode(erosion,kernel,iterations =1 )
#erosion = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel)

dilate = cv2.dilate(img,kernel,iterations = 1)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


cv2.imshow('erosion',erosion)


img=erosion
img1 = cv2.imread(filenama)
print(img)
h=img.shape[0]
l=img.shape[1]
print(img[0][0])
for i in range(h):
    for j in range(l):
#        print(img[i][j])
        if img[i][j]==255:
            img[i][j]=0
        else:
            img[i][j]=255
#img = cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_CONSTANT)
cv2.imshow('img',img)
print(img)

# Set up the detector with default parameters.
params = cv2.SimpleBlobDetector_Params()
params.filterByCircularity = True
params.minCircularity = 0

params.filterByConvexity = True
params.minConvexity = 0
            
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs.
keypoints = detector.detect(img)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(img1, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
print(len(keypoints))
print('keypoints:',keypoints)


x=int(keypoints[0].pt[0])
y=int(keypoints[0].pt[1])


img = cv2.putText(im_with_keypoints, "L", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)


num=len(keypoints)
img = cv2.putText(im_with_keypoints, "total count:"+str(num), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()