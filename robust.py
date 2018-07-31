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
# from Similar_Model1 import Similar_Model1
# from Similar_Model2 import Similar_Model2
# from Similar_Model3 import Similar_Model3
# from Similar_Model4 import Similar_Model4
# from Similar_Model5 import Similar_Model5
# from Similar_Model6 import Similar_Model6
# from Similar_Model7 import Similar_Model7
# from Similar_Model8 import Similar_Model8
# from Similar_Model9 import Similar_Model9
# from Similar_Model10 import Similar_Model10
# from Similar_Model11 import Similar_Model11
# from Similar_Model12 import Similar_Model12
# from Similar_Model13 import Similar_Model13
# from Similar_Model14 import Similar_Model14
# from Similar_Model15 import Similar_Model15
# from Similar_Model16 import Similar_Model16
# from Similar_Model17 import Similar_Model17
# from Similar_Model18 import Similar_Model18
# from Similar_Model19 import Similar_Model19
# from Similar_Model20 import Similar_Model20
# from Similar_Model21 import Similar_Model21
# from Similar_Model22 import Similar_Model22
# from Similar_Model23 import Similar_Model23
# from Similar_Model24 import Similar_Model24
# from Similar_Model25 import Similar_Model25
# from Similar_Model26 import Similar_Model26
# from Similar_Model27 import Similar_Model27
# from Similar_Model28 import Similar_Model28
# from Similar_Model29 import Similar_Model29
# from Similar_Model30 import Similar_Model30
from utils import *
import csv
import time
import os

start = time.clock()

parser = argparse.ArgumentParser(description='Main function for difference-inducing input generation in MNIST dataset')
parser.add_argument('model',
                    help="the target Model")
parser.add_argument('retrain_num',
                    help="the num of retrain epochs", type=int)
args = parser.parse_args()

retrain_num = args.retrain_num
infnorm_distance = 0.5
seeds = 10000
threshold = 0
grad_iterations = 100
step = 1

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

if args.model == 'Model1':
    model = Model1(input_tensor=input_tensor)
elif args.model == 'Model2':
    model = Model2(input_tensor=input_tensor, retrain_num=retrain_num)
elif args.model == 'Model3':
    model = Model3(input_tensor=input_tensor, retrain_num=retrain_num)
elif args.model == 'Model4':
    model = Model4(input_tensor=input_tensor, retrain_num=retrain_num)
elif args.model == 'Model5':
    model = Model5(input_tensor=input_tensor, retrain_num=retrain_num)

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
# elif args.model == 'Similar_Model11':
#     model = Similar_Model11(input_tensor=input_tensor)
# elif args.model == 'Similar_Model12':
#     model = Similar_Model12(input_tensor=input_tensor)
# elif args.model == 'Similar_Model13':
#     model = Similar_Model13(input_tensor=input_tensor)
# elif args.model == 'Similar_Model14':
#     model = Similar_Model14(input_tensor=input_tensor)
# elif args.model == 'Similar_Model15':
#     model = Similar_Model15(input_tensor=input_tensor)
# elif args.model == 'Similar_Model16':
#     model = Similar_Model16(input_tensor=input_tensor)
# elif args.model == 'Similar_Model17':
#     model = Similar_Model17(input_tensor=input_tensor)
# elif args.model == 'Similar_Model18':
#     model = Similar_Model18(input_tensor=input_tensor)
# elif args.model == 'Similar_Model19':
#     model = Similar_Model19(input_tensor=input_tensor)
# elif args.model == 'Similar_Model20':
#     model = Similar_Model20(input_tensor=input_tensor)
# elif args.model == 'Similar_Model21':
#     model = Similar_Model21(input_tensor=input_tensor)
# elif args.model == 'Similar_Model22':
#     model = Similar_Model22(input_tensor=input_tensor)
# elif args.model == 'Similar_Model23':
#     model = Similar_Model23(input_tensor=input_tensor)
# elif args.model == 'Similar_Model24':
#     model = Similar_Model24(input_tensor=input_tensor)
# elif args.model == 'Similar_Model25':
#     model = Similar_Model25(input_tensor=input_tensor)
# elif args.model == 'Similar_Model26':
#     model = Similar_Model26(input_tensor=input_tensor)
# elif args.model == 'Similar_Model27':
#     model = Similar_Model27(input_tensor=input_tensor)
# elif args.model == 'Similar_Model28':
#     model = Similar_Model28(input_tensor=input_tensor)
# elif args.model == 'Similar_Model29':
#     model = Similar_Model29(input_tensor=input_tensor)
# elif args.model == 'Similar_Model30':
#     model = Similar_Model30(input_tensor=input_tensor)

robust_sample_num = 100
x_train_more = [x_train[0]] * robust_sample_num
y_train_more = [y_train[0]] * robust_sample_num

nc = []
process_num = 0
flag = 0
sum_inf = 0
sum_nc = 0
k = 0

for num in range(seeds):
    orig_img = np.expand_dims(x_train[num], axis=0)
    gen_img = orig_img.copy()
    orig_label = y_train[num]
# dir = './results2/'
# for fn in os.listdir(dir):
#     if 'orig' in fn:
#         img = imread(dir+fn)
#         img = img.astype('float32')
#         img /= 255
#         img = img.reshape((28, 28, 1))
#         orig_img = np.expand_dims(img, axis=0)
#         gen_img = orig_img.copy()
#         if fn[5].isdigit():
#             orig_label = int(fn[5])
#         else:
#             orig_label = int(fn[4])
#     else:
#         continue
    label = np.argmax(model.predict(orig_img)[0])
    if not label == orig_label:
        # model_layer_dict = init_coverage_tables(model)
        # update_coverage(orig_img, model, model_layer_dict)
        # sum_nc += neuron_covered(model_layer_dict)[2]
        #
        # sum_inf += 0
        # k += 1
        # with open("result_robust_nc_2.csv", "a+b") as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerows([[args.model, str(epoch), str(0), str(neuron_covered(model_layer_dict)[2])]])
        print("label is not the origin label")
        continue

    temp = model.predict(orig_img)[0].copy()
    temp[label] = 0
    changedlabel = np.argmax(temp)

    loss2 = K.mean(model.get_layer('before_softmax').output[..., changedlabel])
    loss_goal = loss2

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
                if(temp[i] < -infnorm_distance):
                    temp[i] = -infnorm_distance
            gen_img = orig_img + temp.reshape(orig_img.shape)

            gen_img = np.clip(gen_img, 0, 1)
            predictions1 = np.argmax(model.predict(gen_img)[0])

            if not predictions1 == orig_label:
                x_train_more[k] = gen_img.reshape(28, 28, 1)
                y_train_more[k] = orig_label
                # imsave('./results2/' + str(k) + 'th_' + str(predictions1)+ '_' + str(orig_label) + '.png', (x_train_more[k].reshape(28, 28) * 255).astype('uint8'))
                # imsave('./results2/' + str(k) + 'th_' + str(predictions1)+ '_' + str(orig_label) + '_orig.png', (orig_img.reshape(28, 28) * 255).astype('uint8'))

                infnorm_distance /= 2
                break

            if iters == grad_iterations - 1:
                flag = 1
    if infnorm_distance < 0.5:
        # print(infnorm_distance)
        # imsave('test.png', gen_img.reshape(28, 28))
        # imsave('./results2/' + str(k) + 'th_' + str(predictions1) + '_' + str(orig_label) + '.png',
        #        (x_train_more[k].reshape(28, 28) * 255).astype('uint8'))
        # imsave('./results2/' + str(k) + 'th_' + str(predictions1) + '_' + str(orig_label) + '_orig.png',
        #        (orig_img.reshape(28, 28) * 255).astype('uint8'))
        sum_inf += infnorm_distance
        model_layer_dict = init_coverage_tables(model)
        update_coverage(orig_img, model, model_layer_dict)
        sum_nc += neuron_covered(model_layer_dict)[2]
        k += 1
        # with open("result_robust_nc_2.csv", "a+b") as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerows([[args.model, str(epoch), str(args.infnorm_distance), str(neuron_covered(model_layer_dict)[2])]])
    else:
        print('not find the adversarial sample')

    infnorm_distance = 0.5
    flag = 0

    if k > robust_sample_num-1:
        break
print(k)
inf_mean = sum_inf / k
nc_mean = sum_nc / k

print(inf_mean)
print(nc_mean)
with open('robust_nc_add_retrain.csv', 'a+b') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows([[str(args.model) + '_' + str(retrain_num), str(inf_mean), str(nc_mean)]])
    csvfile.close()

time = str(time.clock() - start)
print("Time used: " + time)

if args.model == "Model1":
    re_train_acc = Model1(input_tensor=input_tensor, train=True, re_train=True, x_train_more=x_train_more,
                          y_train_more=y_train_more)
elif args.model == "Model2":
    re_train_acc = Model2(input_tensor=input_tensor, train=True, re_train=True, x_train_more=x_train_more,
                          y_train_more=y_train_more, retrain_num=retrain_num)
elif args.model == 'Model3':
    re_train_acc = Model3(input_tensor=input_tensor, train=True, re_train=True, x_train_more=x_train_more,
                          y_train_more=y_train_more, retrain_num=retrain_num)
elif args.model == 'Model4':
    re_train_acc = Model4(input_tensor=input_tensor, train=True, re_train=True, x_train_more=x_train_more,
                          y_train_more=y_train_more, retrain_num=retrain_num)
elif args.model == 'Model5':
    re_train_acc = Model5(input_tensor=input_tensor, train=True, re_train=True, x_train_more=x_train_more,
                          y_train_more=y_train_more, retrain_num=retrain_num)

with open('acc.csv', 'a+b') as accfile:
    acc_writer = csv.writer(accfile)
    acc_writer.writerows([[str(args.model) + '_' + str(retrain_num+1), str(re_train_acc)]])
    accfile.close()
