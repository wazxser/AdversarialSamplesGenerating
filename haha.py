# from Model1 import Model1
# from Model2 import Model2
# from Model3 import Model3
#
# from keras.layers import Input
# from scipy.misc import imsave, imread
# import numpy as np
# from keras.datasets import mnist
#
# input_shape = (28, 28, 1)
# input_tensor = Input(input_shape)
#
# model = Model2(input_tensor, train=False)
#
# (x_train, _), (_, _) = mnist.load_data()
# x_train = x_train.astype('float32')
# x_train /= 255
#
# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
#
# # img = imread('./temp/_0_2_82th.png')
# img = x_train[0]
# img = np.expand_dims(img, axis=0)
# res = np.argmax(model.predict(img)[0])
#
# print(res)
from __future__ import print_function
from Model2 import Model2
from keras.layers import Input
from keras.datasets import mnist
from robust_single import robust
import csv
import time
import sys
from utils import *

start = time.clock()

img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_train /= 255
x_test = x_test.astype('float32')
x_test /= 255

input_tensor = Input(shape=input_shape)

model = Model2(input_tensor=input_tensor)
img = x_train[0]
label = y_train[0]
print(robust(model, img, label))
model_layer_dict = init_coverage_tables(model)
update_coverage(model, img, model_layer_dict)
print(neuron_covered(model_layer_dict)[2])