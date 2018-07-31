from __future__ import print_function

import argparse

from keras.datasets import mnist
from keras.layers import Input
from scipy.misc import imsave, imread

from Model3 import Model3
from configs import bcolors
from utils import *

import random
import time

start = time.clock()

parser = argparse.ArgumentParser(description='Main function for difference-inducing input generation in MNIST dataset')
parser.add_argument('transformation', help="realistic transformation type", choices=['light', 'occl', 'blackout'])
parser.add_argument('step', help="step size of gradient descent", type=float)
parser.add_argument('seeds', help="number of seeds of input", type=int)
parser.add_argument('grad_iterations', help="number of iterations of gradient descent", type=int)
parser.add_argument('threshold', help="threshold for determining neuron activated", type=float)
parser.add_argument('b', default=100, type=float)
parser.add_argument('-t', '--target_model', help="target model that we want it predicts differently",
                    choices=[0, 1, 2], default=0, type=int)
parser.add_argument('-sp', '--start_point', help="occlusion upper left corner coordinate", default=(0, 0), type=tuple)
parser.add_argument('-occl_size', '--occlusion_size', help="occlusion size", default=(10, 10), type=tuple)

args = parser.parse_args()

img_rows, img_cols = 28, 28

(x_train, y_train), (_, _) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_train /= 255

input_tensor = Input(shape=input_shape)

model = Model3(input_tensor=input_tensor)

for num in xrange(3000):
    gen_img = np.expand_dims(x_train[num], axis=0)
    orig_img = gen_img.copy()
    orig_label = y_train[num]
    label = np.argmax(model.predict(gen_img)[0])

    if not label == orig_label:
        print(bcolors.OKGREEN + 'input already causes different outputs: {}'.format(label) + bcolors.ENDC)

        gen_img_deprocessed = deprocess_image(orig_img)

        imsave('./results/rand/' + 'already_differ_' + str(label)  + '_' + str(num) + '.png', gen_img_deprocessed)
        continue

    for iters in xrange(args.grad_iterations):
        rand = random.random()
        print(rand)
        grads_value = np.ones_like(gen_img)
        grads_value *= rand
        gen_img = gen_img + grads_value

        temp = gen_img - orig_img
        temp = temp.reshape(28, 28)
        gen_norm = abs(temp).max()
        if gen_norm >= 0.5:
            continue

        gen_img = np.clip(gen_img, 0, 1)
        predictions1 = np.argmax(model.predict(gen_img)[0])
        if not predictions1 == orig_label:
            gen_img_deprocessed = deprocess_image(gen_img)
            orig_img_deprocessed = deprocess_image(orig_img)
            imsave('./results/rand/rand_' + str(predictions1) + '_' + str(num) + '.png',
                   gen_img_deprocessed)
            imsave('./results/rand/rand_' + str(predictions1) + '_'+ str(num) + '_orig.png',
                   orig_img_deprocessed)
            break

print("Time used: " + str(time.clock()-start))
