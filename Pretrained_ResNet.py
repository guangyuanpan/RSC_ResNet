import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
from tensorflow.python.keras import applications
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dropout, Flatten, Dense
from keras.callbacks import Callback,TensorBoard
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# import vgg16
import numpy as np
from numpy import *

import re
import sys
import random
import cv2
import scipy


def img_cate(filename):
    img_width, img_height = 224, 224

    # build the VGG16 network
    base_model = applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3))
    print('Model loaded.')

    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(1000, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1000, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(4, activation='softmax'))

    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
    model.load_weights('./ResNet_trainAll.h5')

    # test on validation
    img_path = str('./uploadFile/') + str(filename)
    Pic = cv2.imread(img_path)
    Pic_regu = scipy.misc.imresize(Pic,[img_height,img_height,3])
    # print(np.array([Pic_regu]).shape)
    FC_predict = model.predict(np.array([Pic_regu]))
    print('\n')
    print('Prediction:')
    print(FC_predict)
    # RC_predict = zeros(4)
    # RC_predict[0] = '%.2f%%' % (FC_predict[0][0] * 100)
    # RC_predict[1] = '%.2f%%' % (FC_predict[0][1] * 100)
    # RC_predict[2] = '%.2f%%' % (FC_predict[0][2] * 100)
    # RC_predict[3] = '%.2f%%' % (FC_predict[0][3] * 100)
    # print(RC_predict)
    Category = 'No Data'
    if np.argmax(FC_predict) == 0:
        Category = 'Bare Pavement'
    if np.argmax(FC_predict) == 1:
        Category = 'Partly Coverage'
    if np.argmax(FC_predict) == 2:
        Category = 'Fully Coverage.'
    if np.argmax(FC_predict) == 3:
        Category = 'Not Recognizable'
    return Category, FC_predict
