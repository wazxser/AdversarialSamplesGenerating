'''
usage: python gen_diff.py -h
'''

from __future__ import print_function

import argparse

from keras.datasets import mnist
from keras.layers import Input
from scipy.misc import imsave, imread

from Model1 import Model1
from Model2 import Model2
from Model3 import Model3
from configs import bcolors
from utils import *

import time
import csv
start = time.clock()

# read the parameter
# argument parsing
parser = argparse.ArgumentParser(description='Main function for difference-inducing input generation in MNIST dataset')
parser.add_argument('transformation', help="realistic transformation type", choices=['light', 'occl', 'blackout'])
parser.add_argument('weight_diff', help="weight hyperparm to control differential behavior", type=float)
parser.add_argument('weight_nc', help="weight hyperparm to control neuron coverage", type=float)
parser.add_argument('step', help="step size of gradient descent", type=float)
parser.add_argument('seeds', help="number of seeds of input", type=int)
parser.add_argument('grad_iterations', help="number of iterations of gradient descent", type=int)
parser.add_argument('threshold', help="threshold for determining neuron activated", type=float)
parser.add_argument('target_model', help="target model that we want it predicts differently",
                    choices=[0, 1, 2], default=2, type=int)
parser.add_argument('-sp', '--start_point', help="occlusion upper left corner coordinate", default=(0, 0), type=tuple)
parser.add_argument('-occl_size', '--occlusion_size', help="occlusion size", default=(10, 10), type=tuple)

args = parser.parse_args()

# input image dimensions
img_rows, img_cols = 28, 28
# the data, shuffled and split between train and test sets
(x_train, _), (_, _) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_train /= 255

# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)

# load multiple models sharing same input tensor
model1 = Model1(input_tensor=input_tensor)
model2 = Model2(input_tensor=input_tensor)
model3 = Model3(input_tensor=input_tensor)

# # init coverage table
model_layer_dict1, model_layer_dict2, model_layer_dict3 = init_coverage_tables_nmodel(model1, model2, model3)

x_train_more = []
y_train_more = []

# ==============================================================================================
# start gen inputs
for num in xrange(args.seeds):
    gen_img = np.expand_dims(x_train[num], axis=0)
    orig_img = gen_img.copy()
    # first check if input already induces differences
    label1, label2, label3 = np.argmax(model1.predict(gen_img)[0]), np.argmax(model2.predict(gen_img)[0]), np.argmax(
        model3.predict(gen_img)[0])

    if not label1 == label2 == label3:
        print(bcolors.OKGREEN + 'input already causes different outputs: {}, {}, {}'.format(label1, label2,
                                                                                            label3) + bcolors.ENDC)

        update_coverage(gen_img, model1, model_layer_dict1, args.threshold)
        update_coverage(gen_img, model2, model_layer_dict2, args.threshold)
        update_coverage(gen_img, model3, model_layer_dict3, args.threshold)

        print(bcolors.OKGREEN + 'covered neurons percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f'
              % (len(model_layer_dict1), neuron_covered(model_layer_dict1)[2], len(model_layer_dict2),
                 neuron_covered(model_layer_dict2)[2], len(model_layer_dict3),
                 neuron_covered(model_layer_dict3)[2]) + bcolors.ENDC)
        averaged_nc = (neuron_covered(model_layer_dict1)[0] + neuron_covered(model_layer_dict2)[0] +
                       neuron_covered(model_layer_dict3)[0]) / float(
            neuron_covered(model_layer_dict1)[1] + neuron_covered(model_layer_dict2)[1] +
            neuron_covered(model_layer_dict3)[
                1])
        print(bcolors.OKGREEN + 'averaged covered neurons %.3f' % averaged_nc + bcolors.ENDC)

    # if all label agrees
    orig_label = label1
    layer_name1, index1 = neuron_to_cover(model_layer_dict1)
    layer_name2, index2 = neuron_to_cover(model_layer_dict2)
    layer_name3, index3 = neuron_to_cover(model_layer_dict3)

    # construct joint loss function
    if args.target_model == 0:
        loss1 = -args.weight_diff * K.mean(model1.get_layer('before_softmax').output[..., orig_label])
        loss2 = K.mean(model2.get_layer('before_softmax').output[..., orig_label])
        loss3 = K.mean(model3.get_layer('before_softmax').output[..., orig_label])
    elif args.target_model == 1:
        loss1 = K.mean(model1.get_layer('before_softmax').output[..., orig_label])
        loss2 = -args.weight_diff * K.mean(model2.get_layer('before_softmax').output[..., orig_label])
        loss3 = K.mean(model3.get_layer('before_softmax').output[..., orig_label])
    elif args.target_model == 2:
        loss1 = K.mean(model1.get_layer('before_softmax').output[..., orig_label])
        loss2 = K.mean(model2.get_layer('before_softmax').output[..., orig_label])
        loss3 = -args.weight_diff * K.mean(model3.get_layer('before_softmax').output[..., orig_label])
    loss1_neuron = K.mean(model1.get_layer(layer_name1).output[..., index1])
    loss2_neuron = K.mean(model2.get_layer(layer_name2).output[..., index2])
    loss3_neuron = K.mean(model3.get_layer(layer_name3).output[..., index3])
    layer_output = (loss1 + loss2 + loss3) + args.weight_nc * (loss1_neuron + loss2_neuron + loss3_neuron)

    # for adversarial image generation
    final_loss = K.mean(layer_output)

    # we compute the gradient of the input picture wrt this loss
    grads = normalize(K.gradients(final_loss, input_tensor)[0])

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_tensor], [loss1, loss2, loss3, loss1_neuron, loss2_neuron, loss3_neuron, grads])

    # we run gradient ascent for 20 steps
    for iters in xrange(args.grad_iterations):
        loss_value1, loss_value2, loss_value3, loss_neuron1, loss_neuron2, loss_neuron3, grads_value = iterate(
            [gen_img])
        if args.transformation == 'light':
            grads_value = constraint_light(grads_value)  # constraint the gradients value
        elif args.transformation == 'occl':
            grads_value = constraint_occl(grads_value, args.start_point,
                                          args.occlusion_size)  # constraint the gradients value
        elif args.transformation == 'blackout':
            grads_value = constraint_black(grads_value)  # constraint the gradients value

        gen_img += grads_value * args.step
        gen_img = np.clip(gen_img, 0, 1)
        predictions1 = np.argmax(model1.predict(gen_img)[0])
        predictions2 = np.argmax(model2.predict(gen_img)[0])
        predictions3 = np.argmax(model3.predict(gen_img)[0])
        if not predictions1 == predictions2 == predictions3:
            update_coverage(gen_img, model1, model_layer_dict1, args.threshold)
            update_coverage(gen_img, model2, model_layer_dict2, args.threshold)
            update_coverage(gen_img, model3, model_layer_dict3, args.threshold)

            print(bcolors.OKGREEN + 'covered neurons percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f'
                  % (len(model_layer_dict1), neuron_covered(model_layer_dict1)[2], len(model_layer_dict2),
                     neuron_covered(model_layer_dict2)[2], len(model_layer_dict3),
                     neuron_covered(model_layer_dict3)[2]) + bcolors.ENDC)
            averaged_nc = (neuron_covered(model_layer_dict1)[0] + neuron_covered(model_layer_dict2)[0] +
                           neuron_covered(model_layer_dict3)[0]) / float(
                neuron_covered(model_layer_dict1)[1] + neuron_covered(model_layer_dict2)[1] +
                neuron_covered(model_layer_dict3)[
                    1])
            print(bcolors.OKGREEN + 'averaged covered neurons %.3f' % averaged_nc + bcolors.ENDC)

            # # save the result to disk
            # imsave('./generated_inputs/' + args.transformation + '_' + str(predictions1) + '_' + str(
            #     predictions2) + '_' + str(predictions3) + '.png',
            #        gen_img_deprocessed)
            # imsave('./generated_inputs/' + args.transformation + '_' + str(predictions1) + '_' + str(
            #     predictions2) + '_' + str(predictions3) + '_orig.png',
            #        orig_img_deprocessed)

            if args.target_model == 0:
                if predictions1 != predictions2 and predictions1 != predictions3:
                    x_train_more.append(gen_img.reshape(28, 28))
                    y_train_more.append(orig_label)
                    gen_img_deprocessed = deprocess_image(gen_img)
                    orig_img_deprocessed = deprocess_image(orig_img)

                    # imsave(
                    #     './generated_inputs/' + args.transformation + '_' + str(predictions3) + '_' + str(num) + '.png',
                    #     gen_img_deprocessed)
                    # imsave('./generated_inputs/' + args.transformation + '_' + str(predictions3) + '_' + str(
                    #     num) + '_orig.png',
                    #        orig_img_deprocessed)
            elif args.target_model == 1:
                if predictions2 != predictions1 and predictions2 != predictions3:
                    x_train_more.append(gen_img.reshape(28, 28))
                    y_train_more.append(orig_label)
                    gen_img_deprocessed = deprocess_image(gen_img)
                    orig_img_deprocessed = deprocess_image(orig_img)

                    # imsave(
                    #     './generated_inputs/' + args.transformation + '_' + str(predictions3) + '_' + str(num) + '.png',
                    #     gen_img_deprocessed)
                    # imsave('./generated_inputs/' + args.transformation + '_' + str(predictions3) + '_' + str(
                    #     num) + '_orig.png',
                    #        orig_img_deprocessed)
            else:
                if predictions3 != predictions2 and predictions3 != predictions1:
                    x_train_more.append(gen_img.reshape(28, 28))
                    y_train_more.append(orig_label)
                    gen_img_deprocessed = deprocess_image(gen_img)
                    orig_img_deprocessed = deprocess_image(orig_img)

                    # imsave('./generated_inputs/' + args.transformation + '_' + str(predictions3) + '_' + str(num) + '.png',
                    #        gen_img_deprocessed)
                    # imsave('./generated_inputs/' + args.transformation + '_' + str(predictions3) + '_' + str(num) +'_orig.png',
                    #        orig_img_deprocessed)

            break

    if len(x_train_more) > 99:
        process_num = num
        break

time = str(time.clock() - start)
print("Time used: " + time)
# if args.target_model == 0:
#     re_train_acc = Model1(input_tensor=input_tensor, train=True, re_train=True, x_train_more=x_train_more,
#                           y_train_more=y_train_more)
# elif args.target_model == 1:
#     re_train_acc = Model2(input_tensor=input_tensor, train=True, re_train=True, x_train_more=x_train_more,
#                           y_train_more=y_train_more)
# else:
#     re_train_acc = Model3(input_tensor=input_tensor, train=True, re_train=True, x_train_more=x_train_more,
#                           y_train_more=y_train_more)

with open("results_deepxplore.csv","a+b") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows([[args.target_model, str(args.weight_nc), str(process_num), time]])
