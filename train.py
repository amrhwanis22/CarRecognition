from model import VGG
from keras.preprocessing.image import ImageDataGenerator
import keras
import matplotlib
from Handlers.callbacks import TrainingMonitor
from Handlers.nn import DeeperGooLeNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras import optimizers
import argparse
from keras.models import load_model
import os
from Handlers.callbacks import EpochCheckpoint
#init total number of epochs to be used in decay method and the starting value of optimizer
TOTAL_EPOCHS=100
START_LR=1e-3

#calculating decay value based on number of epoch
def decay(epoch):
    maxEpcohs=TOTAL_EPOCHS
    baseLR=START_LR
    power=1.0
    alpha=baseLR*(1-(epoch/(float(maxEpcohs))))**power
    return alpha


#createing object from keras image generator to read images directly from directory and validation set
trainGen=ImageDataGenerator(horizontal_flip=True,
                            shear_range=0.2,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            rescale=1/255,
                            zoom_range=0.2,
                            rotation_range=20,validation_split=0.2
                            )

val_gen=ImageDataGenerator(rescale=1/255)

#passing images to the method and the target size of image, batch size used
train=trainGen.flow_from_directory('cars_train\\',target_size=(50,50),batch_size=32)
val_data=val_gen.flow_from_directory('cars_train\\',target_size=(50,50),batch_size=32)

#using plots to view learning curve.
figPath = os.path.sep.join(['output',"vggnet_emotion.png"])
jsonPath = os.path.sep.join(['output',"vggnet_emotion.json"])

#adding callbacks to plot every epoch, call deacy method, finally call epoch checkpoint that save model every 5 epochs.
callbacks=[TrainingMonitor(figPath, json_path=jsonPath,start_at=0)
           ,LearningRateScheduler(decay),
           EpochCheckpoint('checkpoints',every=5,startAt=0)]


#creating object from googlenet arch. and compiling it.
gnet=GoogleNet.build(50,50,3,196)
gnet.compile(loss='categorical_crossentropy',
             optimizer=keras.optimizers.Adam(START_LR),
             metrics=['accuracy'])

#fitting model
gnet.fit_generator(train,validation_data=val_data,epochs=150,callbacks=callbacks)



