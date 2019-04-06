# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 20:01:46 2018

@author: wange
"""

import cv2
import numpy as np
img=cv2.imread('text.jpg',0)
#cv2.imshow('image',img)
#cv2.waitKey(0)
kernel=np.ones((5,5),np.uint8)
erosion=cv2.erode(img,kernel,iterations=1)
cv2.imshow('erosion',erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()