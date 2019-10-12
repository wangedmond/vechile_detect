#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 13:43:34 2019

@author: root
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 15:40:44 2018

@author: wange
"""

import cv2
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input


#camera = cv2.VideoCapture('/root/.keras/datasets/video/output1.avi')
camera = cv2.VideoCapture('D:/BaiduNetdiskDownload/20180809125000~2.mp4')


#bs = cv2.createBackgroundSubtractorKNN()
#bs=cv2.createBackgroundSubtractorMOG2()
#bs=cv2.bgsegm.createBackgroundSubtractorCNT()
#bs=cv2.bgsegm.createBackgroundSubtractorGMG()
#bs=cv2.bgsegm.createBackgroundSubtractorGSOC()
#bs=cv2.bgsegm.createBackgroundSubtractorLSBP()
#bs=cv2.bgsegm.createBackgroundSubtractorMOG()
cv2.namedWindow("surveillance",0)



while (camera.isOpened()):
    t1 = cv2.getTickCount()
    ret, frame = camera.read()
    original_frame=frame
    i=i+1

    fgmask = bs.apply(frame)


    # 阈值化
    th = cv2.threshold(fgmask.copy(), 127, 255, cv2.THRESH_BINARY)[1]
    # 通过对前景掩模进行膨胀和腐蚀处理，相当于进行闭运算
    # 开运算：先腐蚀再膨胀
    # 闭运算：先膨胀再腐蚀
    th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
    dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)
    # 轮廓提取
    images,contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#    contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
    t3 = cv2.getTickCount()
    counter = 0
    for c in contours:

        # 对每一个轮廓，如果面积大于阈值500
        if cv2.contourArea(c) > 0:
            # 绘制外包矩形
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)



    if ret == True:
        cv2.imshow("surveillance", frame)
        cv2.imwrite('/root/Documents/pic/other/detection/'+str(i)+'.jpg',frame)
    else:
        break
    if cv2.waitKey(27) & 0xff == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()
