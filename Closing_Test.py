# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 09:52:19 2019

@author: wange
"""

import cv2
import numpy as np
img=cv2.imread('text.jpg',0)
#cv2.imshow('image',img)
#cv2.waitKey(0)
kernel=np.ones((5,5),np.uint8)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
cv2.imshow('closing',closing)
cv2.waitKey(0)
cv2.destroyAllWindows()