# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 12:36:51 2019

@author: wange
"""

import cv2
import os
import glob

image_size=128
#source_path="D:/test_data/test_data/bus"
#target_path="D:/test_data/test_data/bus_resize/"
source_path="D:/test_data/test_data/car"
target_path="D:/test_data/test_data/car_resize/"

if not os.path.exists(target_path):
    os.makedirs(target_path)

i=0
for im_path in glob.glob(os.path.join(source_path, "*")):
    image = cv2.imread(im_path)
    image=cv2.resize(image,(image_size,image_size))
    i=i+1
    cv2.imwrite(target_path+str(i)+".jpg",image)
print("done")
#image_list=os.listdir(source_path)
#
#i=0
#for file in image_list:
#    i=i+1
#    image_source=cv2.imread(source_path+file)
#    image=cv2.resize(image_source,(image_size,image_size))
#    cv2.imwrite(target_path+str(i)+".jpg",image)
#print("done")