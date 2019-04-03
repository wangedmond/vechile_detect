# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 23:22:41 2018

@author: wange
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
filename='D:/PythonText/opencv/text1.jpg'
img = cv2.imread(filename,0)# Initiate STAR detector
star=cv2.xfeatures2d.StarDetector_create()
#star = cv2.FeatureDetector_create("STAR")
# Initiate BRIEF extractor
#brief = cv2.DescriptorExtractor_create("BRIEF")

brief=cv2.xfeatures2d.BriefDescriptorExtractor_create()
# find the keypoints with STAR
kp = star.detect(img,None)
# compute the descriptors with BRIEF
kp, des = brief.compute(img, kp)

print (brief.descriptorSize())
print (des.shape)