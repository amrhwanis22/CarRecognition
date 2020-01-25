from keras.models import load_model
from os import path
import os
import cv2
import imutils
from keras.preprocessing import image
import keras

#loading model from checkpoints
model=load_model('checkpoints/epoch_95.hdf5')
imgPaths=[c for c in os.listdir('cars_test')]

#creating list to hold predictions

preds_class=[]

#iterating over images in test directory to predict each image
for path in imgPaths:
    img=keras.preprocessing.image.load_img('cars_test\\'+path,target_size=(150,150))
    img=keras.preprocessing.image.img_to_array(img)
    img=img.reshape(1,150,150,3).astype('float')
    img/=255
    preds_class.append(model.predict(img).argmax(axis=1))

#writing predictions list to external .txt file
with open('preds.txt','w+') as fout:
    for i in preds_class:
        print(i)
        fout.write(str(i))
        fout.write('\n')
