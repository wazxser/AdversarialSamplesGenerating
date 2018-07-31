from __future__ import print_function
from keras.datasets import mnist
from keras.layers import Input
from scipy.misc import imsave, imread
from Model1 import Model1
from Model2 import Model2
from Model3 import Model3
from Model4 import Model4
from Model5 import Model5
from Similar_Model1 import Similar_Model1
from Similar_Model2 import Similar_Model2
from Similar_Model3 import Similar_Model3
from Similar_Model4 import Similar_Model4
from Similar_Model5 import Similar_Model5
from configs import bcolors
from utils import *
import csv
import os
import argparse

parser = argparse.ArgumentParser(description='Main function for difference-inducing input generation in MNIST dataset')
parser.add_argument('model',
                    help="the target Model")

img_rows, img_cols = 28, 28
(x_train, _), (_, _) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

input_shape = (img_rows, img_cols, 1)
input_tensor = Input(shape=input_shape)
args = parser.parse_args()

if args.model == 'Model1':
    model = Model1(input_tensor=input_tensor)
elif args.model == 'Model2':
    model = Model2(input_tensor=input_tensor)
elif args.model == 'Model3':
    model = Model3(input_tensor=input_tensor)
elif args.model == 'Model4':
    model = Model4(input_tensor=input_tensor)
else:
    model = Model5(input_tensor=input_tensor)

model_layer_dict = init_coverage_tables(model)

i = 0
sum_nc = 0
while i < 10:
    # img = x_train[60000 - i - 1]
    img = x_train[i]
    img = img.astype('float32')
    img /= 255
    img = img.reshape(1, 28, 28, 1)

    update_coverage(img, model, model_layer_dict, 0)
    # print('%.3f' % neuron_covered(model_layer_dict)[2])
    # sum_nc += neuron_covered(model_layer_dict)[2]
    with open("results_neuron_cov_add.csv", "a+b") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows([[args.model, str(neuron_covered(model_layer_dict)[2])]])
    i += 1
print(neuron_covered(model_layer_dict)[2])