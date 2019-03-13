# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 13:44:04 2019

@author: wange
"""

import cv2
import numpy as np
import PIL
 #T
# 运行之前，检查cascade文件路径是否在相应的目录下
#face_cascade = cv2.CascadeClassifier('D:/PythonText/opencv/xml/vehicle_detection_haarcascades-master/vehicle_detection_haarcascades-master/cars.xml')
eye_cascade = cv2.CascadeClassifier('D:/Users/wange/Anaconda3/envs/tensor/Library/etc/haarcascades/haarcascade_eye.xml')
 
# 读取图像
filename='D:/PythonText/opencv/xml/vehicle_detection_haarcascades-master/vehicle_detection_haarcascades-master/dataset/pic/10.jpg'
img=cv2.imread(filename);
img1=cv2.imread(filename);
#img =cv2.imread('timg.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(gray.shape)
print(gray.shape[0])
ww,hh=gray.shape
print(w,h)
# 检测脸部
img1[0:ww,0:hh]=0;
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
print('Detected ', len(faces), " car")
 
count=1
for (x, y, w, h) in faces:
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y: y + h, x: x + w]
    roi_color = img[y: y + h, x: x + w]
    cv2.imwrite('D:/roi_color'+str(count)+'.jpg',roi_color)
    cv2.imshow('D:/roi_color'+str(count)+'.jpg',roi_color)
    count=count+1
#    img[y: y + h, x: x + w]=0;

    img1[y: y + h, x: x + w]=img[y: y + h, x: x + w]                
#    cv2.imshow('roi_color',roi_color)
#    cv2.imshow('img', img)
#    eyes = eye_cascade.detectMultiScale(roi_gray)
#    for(ex, ey, ew, eh) in eyes:
#        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
#cv2.imshow('roi',roi_gray)
#cv2.imwrite('D:/roi_color.jpg',roi_color)
#cv2.imshow('roi_color',roi_color)
cv2.imwrite('D:/img1.jpg',img1)
cv2.imshow('img1', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()