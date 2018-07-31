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
from utils import *
import csv
import time

start = time.clock()


def robust(model=None, input=None, orig_label=None):
    flag = 0
    infnorm_distance = 0.5
    grad_iterations = 100
    step = 1

    input_shape = (28, 28, 1)
    input_tensor = Input(shape=input_shape)

    orig_img = np.expand_dims(input, axis=0)
    gen_img = orig_img.copy()
    label = np.argmax(model.predict(orig_img)[0])

    if not label == orig_label:
        return None

    temp = model.predict(orig_img)[0].copy()
    temp[label] = 0
    changedlabel = np.argmax(temp)

    loss_goal = K.mean(model.get_layer('before_softmax').output[..., changedlabel])

    final_loss = K.mean(loss_goal)
    # print(K.gradients(final_loss, input_tensor)[0])
    grads = normalize(K.gradients(final_loss, input_tensor)[0])
    iterate = K.function([input_tensor], [loss_goal, grads])

    while infnorm_distance != 0 and flag == 0:
        for iters in range(grad_iterations):
            loss_goal, grads_value = iterate([gen_img])

            gen_img = gen_img + grads_value * step
            temp = gen_img - orig_img
            temp = temp.reshape(-1)

            for i in range(temp.shape[0]):
                if(temp[i] > infnorm_distance):
                    temp[i] = infnorm_distance
                if(temp[i] < infnorm_distance):
                    temp[i] = infnorm_distance
            gen_img = orig_img + temp.reshape(orig_img.shape)

            gen_img = np.clip(gen_img, 0, 1)
            predictions1 = np.argmax(model.predict(gen_img)[0])

            if not predictions1 == orig_label:
                # imsave('./results2/' + str(k) + 'th_' + str(predictions1)+ '_' + str(orig_label) + '.png', (x_train_more[k].reshape(28, 28) * 255).astype('uint8'))
                # imsave('./results2/' + str(k) + 'th_' + str(predictions1)+ '_' + str(orig_label) + '_orig.png', (orig_img.reshape(28, 28) * 255).astype('uint8'))

                infnorm_distance /= 2
                break

            if iters == grad_iterations - 1:
                flag = 1
    print(infnorm_distance)
    return infnorm_distance
    # imsave('test.png', gen_img.reshape(28, 28))
