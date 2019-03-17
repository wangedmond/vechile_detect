# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 20:26:06 2019

@author: wange
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 19:33:15 2019

@author: wange
"""

import cv2
import os
from sklearn import linear_model, tree, svm, neighbors, ensemble
from itertools import chain
import numpy as np
import glob
filename1='D:/PythonText/compare_test/vehicles_GIT/vehicles/GTI_Far/image0074.png'
image1 = cv2.imread(filename1,0)

pos_im_path='D:/PythonText/compare_test/vehicles_GIT/vehicles/GTI_Far'
neg_im_path='D:/PythonText/compare_test/non-vehicles_GTI/non-vehicles/GTI'
test_im_path='D:/PythonText/compare_test/vehicles_GIT/vehicles/Far_test'
winSize = (64,64)
h_blockSize = (8,16)
v_blockSize = (16,8)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9
winStride = (8,8)
padding = (0,0)





fds=[]
labels=[]
x_lable=np.array([1])
y_lable=np.array([0])
for im_path in glob.glob(os.path.join(pos_im_path, "*")):
    image = cv2.imread(im_path,0)
    h_descripter = cv2.HOGDescriptor(winSize,h_blockSize,blockStride,cellSize,nbins)
    v_descripter = cv2.HOGDescriptor(winSize,v_blockSize,blockStride,cellSize,nbins)
    h_hist = h_descripter.compute(image,winStride,padding)
    v_hist = v_descripter.compute(image,winStride,padding)
    hist = np.concatenate((h_hist, v_hist), axis=0)
    x_train=hist.reshape(1,-1)
    x_train=x_train.reshape(-1)
    fds.append(x_train)
    labels.append(x_lable)
    
    
for im_path in glob.glob(os.path.join(neg_im_path, "*")):
    image = cv2.imread(im_path,0)
    h_descripter = cv2.HOGDescriptor(winSize,h_blockSize,blockStride,cellSize,nbins)
    v_descripter = cv2.HOGDescriptor(winSize,v_blockSize,blockStride,cellSize,nbins)
    h_hist = h_descripter.compute(image,winStride,padding)
    v_hist = v_descripter.compute(image,winStride,padding)
    hist = np.concatenate((h_hist, v_hist), axis=0)
    x_train=hist.reshape(1,-1)
    x_train=x_train.reshape(-1)
    fds.append(x_train)
    labels.append(y_lable)
print(np.array(fds).shape)





clf=svm.SVC()
clf.fit(fds,labels)
count=0
num_pic=0
for im_path in glob.glob(os.path.join(test_im_path, "*")):
    image = cv2.imread(im_path,0)
    h_descripter = cv2.HOGDescriptor(winSize,h_blockSize,blockStride,cellSize,nbins)
    v_descripter = cv2.HOGDescriptor(winSize,v_blockSize,blockStride,cellSize,nbins)
    #compute(img[, winStride[, padding[, locations]]]) -> descriptors

    h_hist = h_descripter.compute(image,winStride,padding)
    v_hist = v_descripter.compute(image,winStride,padding)
    hist = np.concatenate((h_hist, v_hist), axis=0)
    hist=hist.reshape(1,-1)
    Prediction_lable=clf.predict(hist)
    print('Prediction_lable',Prediction_lable)
    if Prediction_lable==[1]:
        count=count+1
    num_pic=num_pic+1
P=count/num_pic
print('count:',count)
print('num_pic:',num_pic)
print('Prediction:',P)
"""
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64
"""
#fds = []
#labels = []

