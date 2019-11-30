#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 01:28:00 2019

@author: root
"""

import os
from PIL import Image
import numpy as np
import keras
from imgaug import augmenters as iaa
# import imgaug as ia
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import layers
from keras.layers import Convolution2D, Flatten, Dropout, MaxPooling2D, Dense, Activation

# path = 'D:/Users/wechat/WeChat Files/wxid_g6jqg3tpzv9022/FileStorage/File/2019-11/train/train_split/'
# path = 'D:/data_set/five_class_test/data/train\mix/'
path = '/root/.keras/datasets/train/'


def get_data_from_filenames(filenames):
    images = []
    for file in filenames:
        image = Image.open(file)
        image = image.convert('RGB')
        image = image.resize((100, 100))
        image = np.array(image)
        images.append(image)

    return np.array(images)


def get_filenames_and_labels(path):
    filenames = os.listdir(path)

    filename_list = []
    labels_list = []
    for filename in filenames:
        filename_list.append(os.path.join(path, filename))
        labels_list.append(float(filename.split('_')[0]))
    return np.array(filename_list), np.array(labels_list)


# a,b = get_filenames_and_labels(path)
# b.shape

def augment_batch(images):
    seq = iaa.Sequential([
        iaa.Crop(px=(0, 2)),  # crop images from each side by 0 to 16px (randomly chosen)
        iaa.Fliplr(0.01),  # 0.5 is the probability, horizontally flip 50% of the images
        # iaa.Sometimes(0.5,
        #               iaa.GaussianBlur(sigma=(0, 1.0))
        #              ),
        iaa.Affine(
            #            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-1, 1),
            shear=(-1, 1))
    ],

        random_order=True)
    images_aug = seq(images=images)
    return images_aug


def generator_shuffle_aug(path, batch_size=16):
    # è¯»åæä»¶å?    image_filenames, labels = get_filenames_and_labels(path)

    # è®¡ç®å¾çæ»æ°éï¼ç¨ä½å¤å®ä¾æ®
    image_filenames, labels = get_filenames_and_labels(path)
    img_number = len(image_filenames)

    idx = 0
    shuffle = np.random.permutation(np.arange(img_number))
    while True:
        #        shuffle = np.random.permutation(np.arange(img_number))

        data_shuffle = image_filenames[shuffle]
        label_shuffle = labels[shuffle]
        if idx + batch_size > img_number:
            idx = 0
        start = idx
        idx = idx + batch_size

        temp_img_name = data_shuffle[start:start + batch_size]
        temp_label = label_shuffle[start:start + batch_size]

        output_images = get_data_from_filenames(temp_img_name)
        output_labels = keras.utils.np_utils.to_categorical(temp_label, 20).astype('float32')
        #        output_labels = keras.utils.np_utils.to_categorical(temp_label, 20)
        output_images_aug = augment_batch(output_images)
        output_images_aug = output_images_aug.astype('float32') / 255.
        # yield temp_img_name, temp_label
        yield output_images_aug, output_labels


gen = generator_shuffle_aug(path=path)
for i in range(1):
    a, b = next(gen)
print(a.shape)
print(b.shape)
# plt.imshow(a[0])
print(b[0])
# print(b)

model = Sequential()

# layer 1
model.add(layers.Convolution2D(
    filters=32,
    kernel_size=[3, 3],
    padding='same',
    input_shape=(100, 100, 3)
))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(
    pool_size=(2, 2),
    strides=(2, 2),
    padding='same',
))

# layer 2
model.add(Convolution2D(
    filters=64,
    kernel_size=[3, 3],
    padding='same',
))
model.add(Activation('relu'))
model.add(MaxPooling2D(
    pool_size=(2, 2),
    strides=(2, 2),
    padding='same',

))

# layer 3
model.add(Convolution2D(
    filters=128,
    kernel_size=[3, 3],
    padding='same',
))
model.add(Activation('relu'))
model.add(MaxPooling2D(
    pool_size=(2, 2),
    strides=(2, 2),
    padding='same',
))

# fully connected layer 1
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

model.add(Dense(512))
model.add(Activation('relu'))

model.add(Dense(256))
model.add(Activation('relu'))

model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dense(20))
model.add(Activation('softmax'))

model.summary()

adam = keras.optimizers.Adam(lr=0.0001)

model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )
# model.compile(loss='categorical_crossentropy',
#              # optimizer=keras.optimizers.Adam(lr=0.001),
#              optimizer=keras.optimizers.SGD(lr=0.001),
#              metrics=['acc']
#              )

model.fit_generator(
    #        Generator(path).train(),
    generator_shuffle_aug(path=path, batch_size=16),
    # generator_shuffle(path=path),
    # Generator(path).train(),
    steps_per_epoch=200,
    epochs=20,
    # shuffle=False
)

names = os.listdir(path)
print(names[0])
print(names[1000])
images = [os.path.join(path, name) for name in names]
images = get_data_from_filenames(images)
predictions = model.predict(images)
print(predictions[0].argmax())
print(predictions[1000].argmax())