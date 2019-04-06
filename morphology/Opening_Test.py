# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 09:50:56 2019

@author: wange
"""

import cv2
import numpy as np
img=cv2.imread('text.jpg',0)
#cv2.imshow('image',img)
#cv2.waitKey(0)
kernel=np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
cv2.imshow('opening',opening)
cv2.waitKey(0)
cv2.destroyAllWindows()