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
from sklearn import linear_model, tree, svm, neighbors, ensemble
from itertools import chain
import numpy as np

filename='D:/PythonText/compare_test/vehicles_GIT/vehicles/GTI_Far/image0072.png'
filename1='D:/PythonText/compare_test/non-vehicles_GTI/non-vehicles\GTI/image1.png'
filename2='D:/PythonText/compare_test/vehicles_GIT/vehicles/GTI_Far/image0074.png'
image = cv2.imread(filename,0)
image1 = cv2.imread(filename1,0)
image2 = cv2.imread(filename2,0)
winSize = (64,64)
h_blockSize = (8,16)
v_blockSize = (16,8)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9
winStride = (8,8)
padding = (0,0)

"""
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64
"""
fds = []
labels = []

h_descripter = cv2.HOGDescriptor(winSize,h_blockSize,blockStride,cellSize,nbins)
v_descripter = cv2.HOGDescriptor(winSize,v_blockSize,blockStride,cellSize,nbins)
#compute(img[, winStride[, padding[, locations]]]) -> descriptors

h_hist = h_descripter.compute(image,winStride,padding)
v_hist = v_descripter.compute(image,winStride,padding)
hist = np.concatenate((h_hist, v_hist), axis=0)

#print(hist.shape)
#print(hist)
#print(hist.reshape(1,-1))

clf=svm.SVC()

x_train=hist.reshape(1,-1)
x_train=x_train.reshape(-1)
fds.append(x_train)
#x_train=hist
print(x_train.shape)
print(x_train)
#x_train=list(chain(*x_train))
#print(x_train)
y_train=np.array([1])
labels.append(y_train)
print(y_train.shape)
print(y_train)
#y_train=1

#clf.fit(x_train,y_train)
#




h_descripter = cv2.HOGDescriptor(winSize,h_blockSize,blockStride,cellSize,nbins)
v_descripter = cv2.HOGDescriptor(winSize,v_blockSize,blockStride,cellSize,nbins)
#compute(img[, winStride[, padding[, locations]]]) -> descriptors

h_hist = h_descripter.compute(image1,winStride,padding)
v_hist = v_descripter.compute(image1,winStride,padding)
hist = np.concatenate((h_hist, v_hist), axis=0)
#x_test=hist
#score = clf.score(x_test, y_test)
#result = clf.predict(x_test)
x_train=hist.reshape(1,-1)
x_train=x_train.reshape(-1)
fds.append(x_train)
y_train=np.array([0])
labels.append(y_train)
print(np.array(fds).shape)
print(np.array(labels).shape)
print(fds)
#X=np.hstack((x_train,xx_train))
#print(X.shape)
#clf.fit(fds,labels)
print(fds[0])
x=x_train.reshape(-1)
print(x)
clf.fit(fds,labels)



h_descripter = cv2.HOGDescriptor(winSize,h_blockSize,blockStride,cellSize,nbins)
v_descripter = cv2.HOGDescriptor(winSize,v_blockSize,blockStride,cellSize,nbins)
#compute(img[, winStride[, padding[, locations]]]) -> descriptors

h_hist = h_descripter.compute(image2,winStride,padding)
v_hist = v_descripter.compute(image2,winStride,padding)
hist = np.concatenate((h_hist, v_hist), axis=0)
hist=hist.reshape(1,-1)
#hist=hist.reshape(-1)
a=clf.predict(hist)
print('a',a)