# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 12:02:02 2019

@author: wange
"""

import cv2
import os
import glob

image_size=128

#source_path="D:/test_data/test_data/bus"
#target_path="D:/test_data/test_data/bus_resize/"

source_path="D:/Test_data/resize_test_data/Bus_pic/bus_resize"
target_path="D:/resize_pic/bus/"



if not os.path.exists(target_path):
    os.makedirs(target_path)

i=0

for im_path in glob.glob(os.path.join(source_path, "*")):
    filepath,fullflname = os.path.split(im_path)
    fullflname=fullflname.split('.')
    fullflname=str(fullflname[0])
    print(fullflname)
    
    image = cv2.imread(im_path)
    image=cv2.resize(image,(image_size,image_size))

#    i=i+1

    cv2.imwrite(target_path+fullflname+'.jpg',image)
    
print("done")
