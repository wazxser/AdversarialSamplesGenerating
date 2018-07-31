from __future__ import print_function

import argparse

from keras.datasets import mnist
from keras.layers import Input
from scipy.misc import imsave, imread

from Model3 import Model3
from Model1 import Model1
from configs import bcolors
from utils import *

import time

start = time.clock()

parser = argparse.ArgumentParser(description='Main function for difference-inducing input generation in MNIST dataset')
parser.add_argument('transformation', help="realistic transformation type", choices=['light', 'occl', 'blackout'])
parser.add_argument('step', help="step size of gradient descent", type=float)
parser.add_argument('seeds', help="number of seeds of input", type=int)
parser.add_argument('grad_iterations', help="number of iterations of gradient descent", type=int)
parser.add_argument('threshold', help="threshold for determining neuron activated", type=float)
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

model_layer_dict = init_coverage_tables(model)

y_goal = 0

for num in xrange(100):
    model_layer_dict = init_coverage_tables(model)

    orig_img = np.expand_dims(x_train[num], axis=0)
    gen_img = orig_img.copy()
    orig_label = y_train[num]
    label = np.argmax(model.predict(orig_img)[0])

    if not label == orig_label:
        print(bcolors.OKGREEN + 'input already causes different outputs: {}'.format(label) + bcolors.ENDC)

        update_coverage(gen_img, model, model_layer_dict, args.threshold)
        print(bcolors.OKGREEN + 'averaged covered neurons %.3f' % neuron_covered(model_layer_dict)[2] + bcolors.ENDC)

        gen_img_deprocessed = deprocess_image(orig_img)

        imsave('./results/' + args.transformation + '/' + 'already_differ_' + str(label)  + '_'+ str(num) + '.png', gen_img_deprocessed)
        continue

    if y_goal == label:
        y_goal += 1
        if y_goal > 9:
            y_goal %= 10

    loss1 = K.mean(model.get_layer('before_softmax').output[..., label])
    loss2 = K.mean(model.get_layer('before_softmax').output[..., y_goal])
    loss_goal = loss2 - loss1

    layer_name1, index1 = neuron_to_cover(model_layer_dict)
    loss1_neuron = K.mean(model.get_layer(layer_name1).output[..., index1])
    layer_output = loss_goal + loss1_neuron
    final_loss = K.mean(layer_output)
    grads = normalize(K.gradients(final_loss, input_tensor)[0])
    iterate = K.function([input_tensor], [loss1_neuron, grads])

    for iters in xrange(args.grad_iterations):
        loss_neuron1, grads_value = iterate([gen_img])
        if args.transformation == 'light':
            grads_value = constraint_light(grads_value)  # constraint the gradients value
        elif args.transformation == 'occl':
            grads_value = constraint_occl(grads_value, args.start_point,
                                          args.occlusion_size)  # constraint the gradients value
        elif args.transformation == 'blackout':
            grads_value = constraint_black(grads_value)  # constraint the gradients value

        gen_img = gen_img + grads_value * args.step
        imsave('gen_img.png', gen_img.reshape(28, 28))
        temp = gen_img - orig_img
        temp = temp.reshape(28, 28)
        gen_norm = abs(temp).max()
        if gen_norm >= 0.5:
            continue

        # gen_img = np.clip(gen_img, 0, 1)
        predictions1 = np.argmax(model.predict(gen_img)[0])
        if not predictions1 == orig_label:
            update_coverage(gen_img, model, model_layer_dict, args.threshold)
            print(bcolors.OKGREEN + 'averaged covered neurons %.3f' % neuron_covered(model_layer_dict)[2] + bcolors.ENDC)

            gen_img_deprocessed = deprocess_image(gen_img)
            orig_img_deprocessed = deprocess_image(orig_img)
            imsave('./results/' + args.transformation + '/' + args.transformation + '_' + str(predictions1) + '_' + str(num) + '.png',
                   gen_img_deprocessed)
            # imsave('./results/' + args.transformation + '/' + args.transformation + '_' + str(predictions1) + '_'+ str(num) + '_orig.png',
            #        orig_img_deprocessed)
            break

print("Time used: " + str(time.clock()-start))
