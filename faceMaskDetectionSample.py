#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 10:14:42 2022

@author: khusbububna

Final code for Mask No mask cnn Model
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.preprocessing.image import ImageDataGenerator

cnn = Sequential([Conv2D(filters=100, kernel_size=(3,3), 
                    activation='relu'),
                    MaxPooling2D(pool_size=(2,2)),
                    Conv2D(filters=100, kernel_size=(3,3), 
                    activation='relu'),
                    MaxPooling2D(pool_size=(2,2)),
                    Flatten(),
                    Dropout(0.5),
                    Dense(50),
                    Dense(35),
                    Dense(2)])
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])


TRAINING_DIR = "./FaceMaskDataset/Train"
train_datagen = ImageDataGenerator(rescale=1.0/255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(TRAINING_DIR, 
                                                    batch_size=10, 
                                                    target_size=(150, 150))
VALIDATION_DIR = "./FaceMaskDataset/Validation"
validation_datagen = ImageDataGenerator(rescale=1.0/255)
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR, 
                                                         batch_size=10, 
                                                         target_size=(150, 150))

history = cnn.fit_generator(train_generator,
                              epochs=2,
                              validation_data=validation_generator)
#history = cnn.fit_generator(train_generator,
#                            epochs=2)

cnn.save('./FaceDetectionmodelcnn.h5')