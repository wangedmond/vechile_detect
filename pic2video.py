# -*- coding: utf-8 -*-
"""
Created on Tue May 21 10:31:00 2019

@author: wange
"""

#import cv2
#cap=cv2.VideoCapture('D:/data_set/Highway/video1.avi')
##以下三种都可以,参数为保存的文件名，编码格式，帧率，图片大小，色彩模式（默认为FALSE灰度图片）
#
width=320
height=240
#writer1=cv2.VideoWriter('D:/myresult.avi',cv2.VideoWriter_fourcc(*'DIVX'),cap.get(cv2.CAP_PROP_FPS),(width,height),True)
#writer2=cv2.VideoWriter('D:/myresult.avi',cv2.VideoWriter_fourcc(*'MJPG'),cap.get(cv2.CAP_PROP_FPS),(width,height),True)
#writer3=cv2.VideoWriter('D:/myresult.mp4',cv2.VideoWriter_fourcc(*'MP42'),cap.get(cv2.CAP_PROP_FPS),(width,height),True)
#for i in range(60*round(cap.get(cv2.CAP_PROP_FPS))):#截取1分钟
#    _,frame=cap.read()
#    writer1.write(frame)
#cap.release()
#writer1.release()



import numpy as np
import cv2
import glob
import os
from keras.preprocessing import image


#cap = cv2.VideoCapture('D:/data_set/Highway/video1.avi')


file_path='D:/data_set/DETRAC-train-data/Insight-MVT_Annotation_Train/MVI_40992'
video_name=file_path.split('/')[-1]
width=960
height=540
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('D:/'+str(video_name)+'.avi',fourcc, 25.0, (width,height))
f_names=sorted(glob.glob(os.path.join(file_path,'*jpg')))
for i in range(len(f_names)):
    images = cv2.imread(f_names[i])
    frame=images
    #此时不能直接使用out.write(frame)，图片格式可能不一样
    out.write(np.uint8(frame))
#print(f_names)
out.release()
cv2.destroyAllWindows()

    
    
    
    
    
    
## Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('D:/output.avi',fourcc, 25.0, (width,height))
#
#while(cap.isOpened()):
#    ret, frame = cap.read()
#    if ret==True:
##        frame = cv2.flip(frame,0)
#
#        # write the flipped frame
#        out.write(frame)
#
#        cv2.imshow('frame',frame)
#        if cv2.waitKey(1) & 0xFF == ord('q'):
#            break
#    else:
#        break
#
## Release everything if job is finished
#cap.release()
#out.release()
#cv2.destroyAllWindows()
