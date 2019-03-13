# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 21:43:47 2019

@author: wange
"""

import cv2
import numpy as np

img=cv2.imread('D:/PythonText/Harr_HOG_Detection/1.jpg',0);

img1=np.power(img/float(np.max(img)), 1/1.5)
img2 = np.power(img/float(np.max(img)), 1.5)

cv2.imshow('src',img)
cv2.imshow('gamma=1/1.5',img1)
cv2.imshow('gamma=1.5',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()