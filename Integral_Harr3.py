# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 09:52:25 2019

@author: wange
"""

import cv2
import numpy as np
 #T
# 运行之前，检查cascade文件路径是否在相应的目录下
face_cascade = cv2.CascadeClassifier('D:/PythonText/opencv/xml/vehicle_detection_haarcascades-master/vehicle_detection_haarcascades-master/cars.xml')
#eye_cascade = cv2.CascadeClassifier('D:/Users/wange/Anaconda3/envs/tensor/Library/etc/haarcascades/haarcascade_eye.xml')
 
# 读取图像
img=cv2.imread('D:/PythonText/opencv/xml/vehicle_detection_haarcascades-master/vehicle_detection_haarcascades-master/dataset/pic/10.jpg');
#img =cv2.imread('timg.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
# 检测脸部
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
print('Detected ', len(faces), " car")
 
for (x, y, w, h) in faces:
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y: y + h, x: x + w]
    roi_color = img[y: y + h, x: x + w]
    
#    eyes = eye_cascade.detectMultiScale(roi_gray)
#    for(ex, ey, ew, eh) in eyes:
#        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
 
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()