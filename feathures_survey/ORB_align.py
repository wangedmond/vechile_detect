# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 18:06:13 2018

@author: wange
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

filename1='D:/PythonText/opencv/text1.jpg'
filename='D:/PythonText/opencv/text.jpg'
img1 = cv2.imread(filename,0)
img2 = cv2.imread(filename,0)

#Initiate SIFT detector
orb=cv2.ORB_create()

#find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

bf = cv2.BFMatcher()

matches = bf.knnMatch(des1,des2, k=2)

#matches = sorted(matches, key = lambda x:x.distance)

good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
plt.imshow(img3),plt.show()