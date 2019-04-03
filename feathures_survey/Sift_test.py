# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 22:20:58 2018

@author: wange
"""

import cv2
import numpy as np

filename='D:/PythonText/opencv/text1.jpg'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp=sift.detect(gray,None)

img=cv2.drawKeypoints(gray,kp,img)

cv2.imwrite('sift_keypoints.jpg',img)