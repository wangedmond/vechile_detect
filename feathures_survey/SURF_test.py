# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 20:23:10 2018

@author: wange
"""

import cv2
from matplotlib import pyplot as plt

filename='D:/PythonText/opencv/text1.jpg'
img = cv2.imread(filename,0)
cv2.imshow('1',img)
if cv2.waitKey(0)&0xff==27:
    cv2.destroyAllWindows()
#img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

surf = cv2.xfeatures2d.SURF_create(400)

kp,des=surf.detectAndCompute(img,None)

len(kp)
Threshold=surf.getHessianThreshold
#Threshold=int('Threshold', 16)
print(surf.getHessianThreshold)

#surf.hessianThreshold=5000
#
#kp,des=surf.detectAndCompute(img,None)
#
#print(len(kp))
img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
plt.imshow(img2),plt.show()