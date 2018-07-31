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
from Similar_Model6 import Similar_Model6
from Similar_Model7 import Similar_Model7
from Similar_Model8 import Similar_Model8
from Similar_Model9 import Similar_Model9
from Similar_Model10 import Similar_Model10
from configs import bcolors
from utils import *
import csv
import os
import argparse

parser = argparse.ArgumentParser(description='Main function for difference-inducing input generation in MNIST dataset')
parser.add_argument('model',
                    help="the target Model")

img_rows, img_cols = 28, 28
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

input_shape = (img_rows, img_cols, 1)
input_tensor = Input(shape=input_shape)

args = parser.parse_args()

if args.model == 'Model1':
    model = Model1(input_tensor=input_tensor)
elif args.model == 'Model2':
    model = Model2(input_tensor=input_tensor, is_retrain=0, epoch=30)
elif args.model == 'Model3':
    model = Model3(input_tensor=input_tensor)
elif args.model == 'Model4':
    model = Model4(input_tensor=input_tensor)
elif args.model == 'Model5':
    model = Model5(input_tensor=input_tensor)

# if args.model == 'Similar_Model1':
#     model = Similar_Model1(input_tensor=input_tensor)
# elif args.model == 'Similar_Model2':
#     model = Similar_Model2(input_tensor=input_tensor)
# elif args.model == 'Similar_Model3':
#     model = Similar_Model3(input_tensor=input_tensor)
# elif args.model == 'Similar_Model4':
#     model = Similar_Model4(input_tensor=input_tensor)
# elif args.model == 'Similar_Model5':
#     model = Similar_Model5(input_tensor=input_tensor)
#
# if args.model == 'Similar_Model6':
#     model = Similar_Model6(input_tensor=input_tensor)
# elif args.model == 'Similar_Model7':
#     model = Similar_Model7(input_tensor=input_tensor)
# elif args.model == 'Similar_Model8':
#     model = Similar_Model8(input_tensor=input_tensor)
# elif args.model == 'Similar_Model9':
#     model = Similar_Model9(input_tensor=input_tensor)
# elif args.model == 'Similar_Model10':
#     model = Similar_Model10(input_tensor=input_tensor)

i = 0
sum_nc = 0
while i < 100:
    img = x_test[10000 - i - 1]
    # img = x_train[i]
    img = img.astype('float32')
    img /= 255
    img = img.reshape(1, 28, 28, 1)
    model_layer_dict = init_coverage_tables(model)
    update_coverage(img, model, model_layer_dict, 0)
    # print(neuron_covered(model_layer_dict))
    # print('%.3f' % neuron_covered(model_layer_dict)[2])
    sum_nc += neuron_covered(model_layer_dict)[2]
    i += 1
    # intermediate_layer_model = Model(inputs=model.input,
    #                                  outputs=[model.get_layer('before_softmax').output])
    # print("           " + str(np.var(intermediate_layer_model.predict(img)[0])))
    # print(sum(intermediate_layer_model.predict(img)[0]))

    # print(np.var(model.predict(img)[0]))
    # with open("results333.csv", "a+b") as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerows([[str(neuron_covered(model_layer_dict)[2]), str(np.var(model.predict(img)[0]))]])
print(sum_nc / 100)