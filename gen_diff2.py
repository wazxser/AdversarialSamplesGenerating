from __future__ import print_function

import argparse

from keras.datasets import mnist
from keras.layers import Input
from scipy.misc import imsave, imread

from Model3 import Model3
from configs import bcolors
from utils import *

import os

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

# (x_train, y_train), (_, _) = mnist.load_data()
x_train, y_train = np.empty((1, 28, 28)), []

dir = './gen_diff_img/'
for fn in os.listdir(dir):
    img = imread(dir+fn)
    x_train = np.insert(x_train, 0, values=img, axis=0)
    y_train = np.insert(y_train, 0, values=int(fn[0]), axis=0)
x_train = np.delete(x_train, -1, axis=0)
print(x_train.shape, y_train.shape)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_train /= 255

input_tensor = Input(shape=input_shape)

model = Model3(input_tensor=input_tensor)

model_layer_dict = init_coverage_tables(model)

for num in xrange(20):
    # rand = random.randint(0, args.seeds)

    gen_img = np.expand_dims(x_train[num], axis=0)
    # gen_img = np.expand_dims(mpimg.imread("./orig.png").reshape(28, 28, 1).astype("float32"), axis=0)
    orig_img = gen_img.copy()
    orig_label = y_train[num]
    # orig_label = 3
    label = np.argmax(model.predict(gen_img)[0])
    model_layer_dict = init_coverage_tables(model)
    if not label == orig_label:
        print(bcolors.OKGREEN + 'input already causes different outputs: {}'.format(label) + bcolors.ENDC)

        update_coverage(gen_img, model, model_layer_dict, args.threshold)
        print(bcolors.OKGREEN + 'averaged covered neurons %.3f' % neuron_covered(model_layer_dict)[2] + bcolors.ENDC)

        gen_img_deprocessed = deprocess_image_orig(orig_img)

        imsave('./results/' + args.transformation + '/' + 'already_differ_' + str(label)  + '_'+ str(num) + '.png', gen_img_deprocessed)
        continue

    layer_name1, index1 = neuron_to_cover(model_layer_dict)

    loss1_neuron = K.mean(model.get_layer(layer_name1).output[..., index1])

    layer_output = loss1_neuron

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

        gen_img += grads_value * args.step
        gen_img = np.clip(gen_img, 0, 1)
        # temp = (grads_value * args.step).reshape(28, 28)
        # gen_2norm = np.linalg.norm(temp, ord=np.inf)

        # if gen_2norm > 18:
        #     continue
        # print(gen_2norm)
        predictions1 = np.argmax(model.predict(gen_img)[0])
        # print(gen_img.shape)
        # imsave("./test.png", gen_img.reshape(28, 28))
        if not predictions1 == orig_label:
            update_coverage(gen_img, model, model_layer_dict, args.threshold)
            print(bcolors.OKGREEN + 'averaged covered neurons %.3f' % neuron_covered(model_layer_dict)[2] + bcolors.ENDC)

            gen_img_deprocessed = deprocess_image(gen_img)
            orig_img_deprocessed = deprocess_image(orig_img)

            imsave('./results/' + args.transformation + '/' + args.transformation + '_' + str(predictions1) + '_' + str(num) + '.png',
                   gen_img_deprocessed)
            imsave('./results/' + args.transformation + '/' + args.transformation + '_' + str(predictions1) + '_'+ str(num) + '_orig.png',
                   orig_img_deprocessed)
            break
