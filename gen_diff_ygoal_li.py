from __future__ import print_function

import argparse

from keras.datasets import mnist
from keras.layers import Input
from scipy.misc import imsave, imread

from Model1 import Model1
from Model2 import Model2
from Model3 import Model3
from Model4 import Model4
from Model5 import Model5
from configs import bcolors
from utils import *
import csv
import time

start = time.clock()

parser = argparse.ArgumentParser(description='Main function for difference-inducing input generation in MNIST dataset')
parser.add_argument('step',
                    help="step size of gradient descent", type=float)
parser.add_argument('grad_iterations',
                    help="number of iterations of gradient descent", type=int)
parser.add_argument('threshold',
                    help="threshold for determining neuron activated", type=float)
parser.add_argument('weights_label',
                    help="the weights for change label", type=float)
parser.add_argument('weights_nc',
                    help="the weights for neuron coverage", type=float)
parser.add_argument('infnorm_distance',
                    help="the inf norm constraint for the distance of gen_img and orig_img", type=float)
parser.add_argument('model',
                    help="the target Model")
parser.add_argument('seeds',
                    help='the numer of the input seeds', type=int)

args = parser.parse_args()

img_rows, img_cols = 28, 28

(x_train, y_train), (_, _) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_train /= 255

input_tensor = Input(shape=input_shape)

if args.model == 'Model1':
    model = Model1(input_tensor=input_tensor)
elif args.model == 'Model2':
    model = Model2(input_tensor=input_tensor)
elif args.model == 'Model3':
    model = Model3(input_tensor=input_tensor)
elif args.model == 'Model4':
    model = Model4(input_tensor=input_tensor)
elif args.model == 'Model5':
    model = Model5(input_tensor=input_tensor)

model_layer_dict = init_coverage_tables(model)

x_train_more = []
y_train_more = []

process_num = 0
for num in range(args.seeds):
    orig_img = np.expand_dims(x_train[num], axis=0)
    gen_img = orig_img.copy()
    orig_label = y_train[num]
    label = np.argmax(model.predict(orig_img)[0])

    if not label == orig_label:
        # print(bcolors.OKGREEN +
        #       'input already causes different outputs: {}'.format(label) + bcolors.ENDC)
        #
        # update_coverage(gen_img, model, model_layer_dict, args.threshold)
        # print(bcolors.OKGREEN + 'averaged covered neurons %.3f' %
        #       neuron_covered(model_layer_dict)[2] + bcolors.ENDC)
        #
        # gen_img_deprocessed = deprocess_image(orig_img)
        #
        # imsave('results/' + 'already_differ_' +
        #        str(label) + '_' + str(num) + '.png', gen_img_deprocessed)
        continue

    temp = model.predict(orig_img)[0].copy()
    temp[label] = 0
    changedlabel = np.argmax(temp)

    loss2 = K.mean(model.get_layer('before_softmax').output[..., changedlabel])
    loss_goal = loss2

    layer_name1, index1 = neuron_to_cover(model_layer_dict)
    loss1_neuron = K.mean(model.get_layer(layer_name1).output[..., index1])
    layer_output = args.weights_label * loss_goal + args.weights_nc * loss1_neuron
    final_loss = K.mean(layer_output)
    grads = normalize(K.gradients(final_loss, input_tensor)[0])
    iterate = K.function([input_tensor], [loss_goal, loss1_neuron, grads])

    for iters in range(args.grad_iterations):
        loss_goal, loss1_neuron, grads_value = iterate([gen_img])

        gen_img = gen_img + grads_value * args.step
        temp = gen_img - orig_img
        temp = temp.reshape(-1)

        for i in range(temp.shape[0]):
            if(temp[i] > args.infnorm_distance):
                temp[i] = args.infnorm_distance
            if(temp[i] < -args.infnorm_distance):
                temp[i] = -args.infnorm_distance
        gen_img = orig_img + temp.reshape(orig_img.shape)
        gen_img = np.clip(gen_img, 0, 1)
        predictions1 = np.argmax(model.predict(gen_img)[0])

        # if not predictions1 == orig_label:
        #     x_train_more.append(orig_img.reshape(28, 28, 1))
        #     y_train_more.append(orig_label)
        #
        #     update_coverage(gen_img, model, model_layer_dict, args.threshold)
        #     print(bcolors.OKGREEN + 'averaged covered neurons %.3f' %
        #           neuron_covered(model_layer_dict)[2] + bcolors.ENDC)
        #
        #     gen_img_deprocessed = deprocess_image(gen_img)
        #     orig_img_deprocessed = deprocess_image(orig_img)
        #     imsave('results2/' + '_' + str(predictions1) + '_' + str(label) + '_' + str(num) + 'th' + '.png',
        #            gen_img_deprocessed)
        #     imsave('results2/' + '_' + str(predictions1) + '_' + str(label) + '_' + str(num) + 'th' + '_orig.png',
        #            orig_img_deprocessed)
        #     break

    # if len(x_train_more) > 99:
    #     process_num = num
    #     break

    gen_img_deprocessed = deprocess_image(gen_img)
    orig_img_deprocessed = deprocess_image(orig_img)
    imsave('results2/' + str(num) + 'th' + '_' + str(label) + '_' + str(label) + '.png',
           gen_img_deprocessed)
    imsave('results2/' + str(num) + 'th' + '_' + str(label) + '_' + str(label) + '_orig.png',
           orig_img_deprocessed)

time = str(time.clock() - start)
print("Time used: " + time)
# if args.model == "Model1":
#     re_train_acc = Model1(input_tensor=input_tensor, train=True, re_train=True, x_train_more=x_train_more,
#                           y_train_more=y_train_more)
# elif args.model == "Model2":
#     re_train_acc = Model2(input_tensor=input_tensor, train=True, re_train=True, x_train_more=x_train_more,
#                           y_train_more=y_train_more)
# else:
#     re_train_acc = Model3(input_tensor=input_tensor, train=True, re_train=True, x_train_more=x_train_more,
#                           y_train_more=y_train_more)
#
# with open("results.csv","a+b") as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows([[args.model, str(args.weights_nc), str(process_num), str(re_train_acc), time]])