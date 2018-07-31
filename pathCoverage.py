import numpy as np
from keras.datasets import mnist
from Model1 import Model1
from Model2 import Model2
from keras.layers import Input
from utils import *
import os
from scipy.misc import imread

(x_train, y_train), (_, _) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

x_train = x_train.astype('float32')
x_train /= 255

model = Model2(input_tensor=Input(shape=(28, 28, 1)))

model_layer_dict = init_coverage_tables(model)

num_path_all = 1
for layer in model.layers:
    if 'flatten' not in layer.name and 'input' not in layer.name:
        num_path_all *= layer.output_shape[-1]
print(num_path_all)

dir = './temp2/'
for fn in os.listdir(dir):
    model_layer_dict = init_coverage_tables(model)
    layer_act_num = init_layer_act_num(model)

    img = imread(dir+fn).reshape(28, 28, 1)
    img = np.expand_dims(img, axis=0)
    update_path_coverage(img, model, model_layer_dict, layer_act_num)
    # print(path_coverged(layer_act_num))
    print(path_coverged(layer_act_num)/ float(num_path_all))
