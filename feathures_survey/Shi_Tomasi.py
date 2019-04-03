# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 21:36:15 2018

@author: wange
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

filename='D:/PythonText/opencv/text1.jpg'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

coeners=cv2.goodFeaturesToTrack(gray,25,0.01,10)
coeners=np.int0(coeners)

for i in coeners:
    x,y=i.ravel()
    cv2.circle(img,(x,y),3,255,-1)

plt.imshow(img)
cv2.imshow('1',img)
if cv2.waitKey(0)&0xff==27:
    cv2.destroyAllWindows()