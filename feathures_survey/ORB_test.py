# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 21:58:00 2018

@author: wange
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

filename='D:/PythonText/opencv/text1.jpg'
img = cv2.imread(filename,0)

orb=cv2.ORB_create()
#kp=orb.detect(img,None)
#
#kp,des=orb.compute(img,kp)
kp,des=orb.detectAndCompute(img,None)


img2=cv2.drawKeypoints(img,kp,img,color=(0,255,0),flags=0)
plt.imshow(img2),plt.show()