from __future__ import print_function
from keras.applications.vgg16 import VGG16
from keras.layers import Input
from scipy.misc import imsave, imread
from keras.preprocessing import image
from utils import *
from keras.applications.vgg16 import preprocess_input, decode_predictions

import os

img_rows, img_cols = 224, 224
input_shape = (img_rows, img_cols, 3)
input_tensor = Input(shape=input_shape)

K.set_learning_phase(0)
model = VGG16(input_tensor=input_tensor)
model_layer_dict = init_coverage_tables(model)

dir = './one_class_img/'
for fn in os.listdir(dir):
    img_path = dir+fn
    img = image.load_img(img_path, target_size=(224, 224))
    input_img_data = image.img_to_array(img)
    input_img_data = np.expand_dims(input_img_data, axis=0)
    input_img_data = preprocess_input(input_img_data)
    update_coverage(input_img_data, model, model_layer_dict, 0)

    print('total neurons: %d' % neuron_covered(model_layer_dict)[1])
    print('covered neurons: %d' % neuron_covered(model_layer_dict)[0])
    print('%.3f' % neuron_covered(model_layer_dict)[2])