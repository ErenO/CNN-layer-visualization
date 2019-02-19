import re
import sys
import math
import collections
import cv2 as cv
import os

import matplotlib.image as mpimg
from matplotlib import pyplot as plt

import tensorflow as tf
import pandas as pd
import numpy as np

import itertools

import keras
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.utils.training_utils import multi_gpu_model
from keras.models import Sequential, Model, load_model
from keras.layers import *
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications import VGG16
from keras.preprocessing import image
from keras import backend as K
from keras import activations

from flask import Flask, flash, render_template, request, session, request, Markup
from flask_bootstrap import Bootstrap

from vis.utils import utils
from vis.visualization import get_num_filters
from vis.visualization import visualize_activation

from functools import partial
import functools

from vis.utils import utils
from vis.visualization import visualize_activation

from PIL import *

# create a new image name, return the path
def img_name(layer, filter_nb):
    path = 'static/img/' + layer + '/'
    imgFolder = os.listdir(path)
    imgName = layer + '_' + str(filter_nb) + '.jpg'    
    if not imgName in imgFolder:
        return (path + imgName)
    else:
        return (None)
        
def layer_visualisation(model, layer, start, end):
    layer_idx = utils.find_layer_idx(model, layer)
    filters = np.random.permutation(get_num_filters(model.layers[layer_idx]))[:10]
    nbFilters = len(filters)
    # create a folder in static/img/ for every layers
    path = "static/img/" + layer + '/'
    if not os.path.isdir(path):
        os.makedirs(path)
    for index in range(start, end + 1):
        if index < nbFilters:
            namePath = img_name(layer, index)
            if not os.path.exists(namePath):
                layer_idx = utils.find_layer_idx(model, layer) #layer
                plt.rcParams['figure.figsize'] = (18, 6)
                img = visualize_activation(model, layer_idx, filter_indices=index) #filter_nb
                im = Image.fromarray(img)
                d = ImageDraw.Draw(im)
                d.text((10,10), layer + " Filter: " + str(index), fill=(255,255,0))
                print (layer + " Filter: " + str(index))
                im.save(namePath)
                print ('created ' + namePath)
            else:
                print (namePath + ' already exists' + str(os.path.exists(namePath)))

# replace the code below by your model architecture
# ====>
def make_model(): 
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=[256, 256, 3])
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(9, activation='softmax')(x)
    with tf.device("/cpu:0"):
        mod = Model(inputs=base_model.input, outputs=predictions)
    print(len(mod.layers))
    return (mod)
# <====

def sumModel(model):
    orig_stdout = sys.stdout
    f = open('out.txt', 'w')
    sys.stdout = f
    print(model.summary())
    sys.stdout = orig_stdout
    f.close()
    file = open('out.txt')
    string = ''
    for line in file:
        string = string + line + '\n'
    return (string)
    
def create_model(name):
    model = make_model()
    inp = [None] * len(model.layers)
    out = [None] * len(model.layers)
    for index, layer in enumerate(model.layers):
        m = re.search("(?<=\")\w+", str(layer.get_input_at(0)))
        if m:
            if "None" not in m.group(0):
                inp[index] = m.group(0)
        m = re.search("(?<=\")\w+", str(layer.get_output_at(0)))
        if m:
            out[index] = m.group(0)
    return (model, inp, out)
