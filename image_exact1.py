# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 18:47:48 2019

@author: wange
"""

import cv2
#T
vc=cv2.VideoCapture('D:/PythonText/opencv/xml/vehicle_detection_haarcascades-master/vehicle_detection_haarcascades-master/dataset/video2.avi')
#c是用来控制取哪帧图像的
c=1
pic_count=1

if vc.isOpened():
    rval,frame=vc.read()
else:
    rval=False

timeF=15

while rval:
    rval,frame=vc.read()
    if(c%timeF==0):
        cv2.imwrite('D:/PythonText/opencv/xml/vehicle_detection_haarcascades-master/vehicle_detection_haarcascades-master/dataset/pic/'+str(pic_count)+'.jpg',frame)
        pic_count=pic_count+1
    c=c+1
    cv2.waitKey(1)
vc.release()