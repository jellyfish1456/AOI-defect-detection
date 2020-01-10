# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 20:04:30 2020

@author: haha8
"""

#import library
from shutil import copyfile, move
from tqdm import tqdm
import h5py
import pandas as pd
import numpy as np
import ast
import datetime as dt
import os
import time
from math import trunc
import tensorflow as tf
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.keras.backend.set_session(sess)
import cv2
import json
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 15
import seaborn as sns
from collections import Counter
from PIL import Image
from collections import defaultdict
from pathlib import Path
import keras
import warnings
from keras import models, layers, optimizers, losses, metrics, regularizers
from keras.layers.core import Dense, Flatten, Dropout
from keras.utils.np_utils import to_categorical
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers import Flatten, Conv2D, MaxPool2D
from keras.models import Sequential
from keras.layers import Flatten, Conv2D, MaxPool2D, Activation, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import  preprocess_input
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense, GlobalAveragePooling2D
from keras.metrics import (categorical_accuracy, categorical_crossentropy,
                           top_k_categorical_accuracy)
from keras.models import Model, load_model
from keras.layers.normalization import BatchNormalization
from keras.applications.vgg16 import VGG16

#check version of tensorflow and test whether uses GPU
print(tf.__version__)
print(tf.test.is_gpu_available())

#import csv file
training_df = pd.read_csv(r"C:\Users\haha8\Downloads\aidea_project\ai_project\train.csv")
training_df.head()
src = "train_images/"
dst = "training_data/"

os.mkdir(dst)
os.mkdir(dst+"0")
os.mkdir(dst+"1")
os.mkdir(dst+"2")
os.mkdir(dst+"3")
os.mkdir(dst+"4")
os.mkdir(dst+"5")

with tqdm(total=len(list(training_df.iterrows()))) as pbar:
    for idx, row in training_df.iterrows():
        pbar.update(0)
        if row["Label"] == 0:
            copyfile(src+row["ID"], dst+"0/"+row["ID"])
        elif row["Label"] == 1:
            copyfile(src+row["ID"], dst+"1/"+row["ID"])
        elif row["Label"] == 2:
            copyfile(src+row["ID"], dst+"2/"+row["ID"])
        elif row["Label"] == 3:
            copyfile(src+row["ID"], dst+"3/"+row["ID"])
        elif row["Label"] == 4:
            copyfile(src+row["ID"], dst+"4/"+row["ID"])
        elif row["Label"] == 5:
            copyfile(src+row["ID"], dst+"5/"+row["ID"])
            
src = "training_data/"
dst = "validation_data/"

os.mkdir(dst)
os.mkdir(dst+"0")
os.mkdir(dst+"1")
os.mkdir(dst+"2")
os.mkdir(dst+"3")
os.mkdir(dst+"4")
os.mkdir(dst+"5")

validation_df = training_df.sample(n=int(len(training_df)/10))

with tqdm(total=len(list(validation_df.iterrows()))) as pbar:
    for idx, row in validation_df.iterrows():
        pbar.update(0)
        if row["Label"] == 0:
            move(src+"0/"+row["ID"], dst+"0/"+row["ID"])
        elif row["Label"] == 1:
            move(src+"1/"+row["ID"], dst+"1/"+row["ID"])
        elif row["Label"] == 2:
            move(src+"2/"+row["ID"], dst+"2/"+row["ID"])
        elif row["Label"] == 3:
            move(src+"3/"+row["ID"], dst+"3/"+row["ID"])
        elif row["Label"] == 4:
            move(src+"4/"+row["ID"], dst+"4/"+row["ID"])
        elif row["Label"] == 5:
            move(src+"5/"+row["ID"], dst+"5/"+row["ID"])

#using data augmentation
batch_size = 32
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range = 0.2,
    width_shift_range = 0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True)

train_data_dir = "training_data"
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    shuffle=True,
    target_size=(100, 100),
    batch_size=50,
    class_mode='categorical')


validation_datagen = ImageDataGenerator(rescale=1. / 255)
validation_data_dir = "validation_data"
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(100, 100),
    batch_size=25,
    class_mode='categorical')

input_shape = (100,100,3)
num_classes = 6

#plot images
sample_training_images, _ = next(train_generator)
def plotImages(images_arr): 
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
plotImages(sample_training_images[:5])

#building the training network. you can change the network you want
model_vgg = VGG16(include_top = False,input_shape=(32,32,3),weights = 'imagenet')
model = Flatten(name = 'flatten')(model_vgg.output)
model = Dense(6,activation='softmax')(model)
model_vgg = Model(model_vgg.input,model,name = 'vgg16')
model_vgg.compile(optimizer =  'adam', loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])  
epochs = 400
history = model_vgg.fit_generator(train_generator,
          validation_data=validation_generator,
          epochs=epochs,
          verbose=1,
          shuffle=False)

# plot model loss & save
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('densenet_try2_Loss_summary_graph.png')
plt.show()

# plot model accuracy & save
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('densenet_try2_Accuracy_summary_graph.png')
plt.show()

# #### Testing on test_data
test_datagen=ImageDataGenerator(rescale=1./255.)
test_generator=test_datagen.flow_from_directory(
        directory=r"C:\Users\haha8\Downloads\aidea_project\test_data",
        target_size=(100,100),
        batch_size=1,
        class_mode='categorical',
        shuffle=False
)
filenames = test_generator.filenames
nb_samples = len(filenames)
predict = model.predict_generator(test_generator,steps = nb_samples, verbose=1)

#sample probability result of one images 
print(predict[0])
label = np.where(predict[0]==max(predict[0]))
label_map = train_generator.class_indices
print(label_map)


#create new file submission for see the result
csv_file = open("densenet.csv","w")
csv_file.write("ID,Label\n")
for filename, prediction in zip(filenames,predict):
    name = filename.split("/")
    name = name[0]
    label = np.where(prediction==max(prediction))
    label = label[0][0]
    csv_file.write(str(name)+","+str(label)+"\n")
csv_file.close()