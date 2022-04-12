#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 10:16:13 2022

@author: khusbububna

Final code for Mas and no Mask
"""

import cv2
import numpy as np
from faceMaskDetectionSample import *
#from tensorflow import keras

#cnn = keras.models.load_model('./FaceDetectionmodelcnn.h5')

labels_dict={0:'WithoutMask',1:'WithMask'}
color_dict={0:(0,0,255),1:(0,255,0)}
imgsize = 4 #set image resize

#camera = cv2.VideoCapture(0) 
camera = cv2.VideoCapture('./input/VideoNigeriaTrimmed.mp4')
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out_video = cv2.VideoWriter('./input/outputFaceMaskVideo.avi', fourcc, 5, (640, 360))
# Identify frontal face
classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    rval, im = camera.read()
    im = cv2.flip(im,1,1) #mirrow the image
    #print(im)
    imgs = cv2.resize(im, (im.shape[1]//imgsize, im.shape[0]//imgsize))
    face_rec = classifier.detectMultiScale(imgs) 
    for i in face_rec: 
        (x, y, l, w) = [v * imgsize for v in i] 
        face_img = im[y:y+w, x:x+l]
        resized=cv2.resize(face_img,(150,150))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,150,150,3))
        reshaped = np.vstack([reshaped])
        result=cnn.predict(reshaped)
        label=np.argmax(result,axis=1)[0]
        #label = "{}: {:.2f}%".format(labels_dict[label], max(mask, withoutMask) * 100)
        cv2.rectangle(im,(x,y),(x+l,y+w),color_dict[label],2)
        cv2.rectangle(im,(x,y-40),(x+l,y),color_dict[label],-1)
        cv2.putText(im, labels_dict[label], (x, y-
        10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    cv2.imshow('LIVE',im)
    #out_video.write(im)
'''
    key = cv2.waitKey(10)

    if key == 27: 
        break
camera.release()
cv2.destroyAllWindows()
'''