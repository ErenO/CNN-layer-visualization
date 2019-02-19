import cv2 as cv
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
import tensorflow as tf
import pandas as pd
import numpy as np
import itertools
from flask import Markup
from flask import Blueprint, Flask, flash, render_template, request, session
from flask_bootstrap import Bootstrap
from vis.utils import utils
from keras import activations
from vis.visualization import get_num_filters
from vis.visualization import visualize_activation
from PIL import Image
import numpy as np
import sys
from keras import backend as K
import visualize as visu
from keras.applications import VGG16
from vis.utils import utils
from keras import activations

app = Flask(__name__, static_url_path='/static')
Bootstrap(app)
app.secret_key = 'my unobvious secret key'
model = None 
inp = None
out = None

@app.before_first_request
def set_model():
    global model
    global inp
    global out
    print ("set_model")
    # change by your model's name
    model, inp, out = visu.create_model('model_name.h5')

@app.route('/handle_data', methods=['GET','POST'])
def handle_data():
    layer = request.form.get('layer_select')
    filter_str = request.form['filter_nb']
    layer_idx = utils.find_layer_idx(model, layer)
    filters = len(np.random.permutation(get_num_filters(model.layers[layer_idx]))[:10])
    if filters > 0:
        if len(filter_str) == 0:
            visu.layer_visualisation(model, layer, 0, filters)
        elif filter_str.isdigit():
            filter_nb = int(filter_str)
            visu.layer_visualisation(model, layer, filter_nb, filter_nb)
    else:
        print ("filters nb : " + filters)
    return (index())


@app.route('/')
def index():
    lst = os.listdir("static/img/")
    lst = [x for x in lst if os.path.isdir('static/img/' + x)]
    dic = {name : [x for x in os.listdir('static/img/' + name)] for name in lst}
    lst = os.listdir("static/img/")
    res = render_template('index.html', 
                       imgFolder=dic,
                       inp_lay=inp,
                       out_lay=out)
    return res

if __name__ == '__main__':
    app.debug = True
    app.run()
    