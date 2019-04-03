# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 21:20:42 2018

@author: wange
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
filename='D:/PythonText/opencv/text1.jpg'
img = cv2.imread(filename,0)
# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create()
# find and draw the keypoints
kp = fast.detect(img,None)
#img2=img.copy()
img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)

cv2.imwrite('fast_true.png',img2)

fast.setNonmaxSuppression(0)
kp = fast.detect(img,None)
img3 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
cv2.imwrite('fast_false.png',img3)




#non-max supression
import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('simple.jpg',0)
# Initiate FAST object with default values
fast = cv2.FastFeatureDetector()
# find and draw the keypoints
kp = fast.detect(img,None)
img2 = cv2.drawKeypoints(img, kp, color=(255,0,0))
# Print all default params
print ("Threshold: ", fast.getInt('threshold'))
print ("nonmaxSuppression: ", fast.getBool('nonmaxSuppression'))
print ("neighborhood: ", fast.getInt('type'))
print ("Total Keypoints with nonmaxSuppression: ", len(kp))
cv2.imwrite('fast_true.png',img2)
# Disable nonmaxSuppression
fast.setBool('nonmaxSuppression',0)
kp = fast.detect(img,None)
print ("Total Keypoints without nonmaxSuppression: ", len(kp))
img3 = cv2.drawKeypoints(img, kp, color=(255,0,0))
cv2.imwrite('fast_false.png',img3)
