import os
from scipy.misc import imread
from Model2 import Model2
from Model3 import Model3
from Model4 import Model4
from keras.layers import Input
import numpy as np

input_tensor = Input(shape=(28, 28, 1))

dir = './results2/'

x_train_more = []
y_train_more = []
model = Model4(input_tensor=input_tensor, is_retrain=1, epoch=1)

for fn in os.listdir(dir):
    if not 'orig' in fn:
        img = imread(dir+fn)
        img = img.astype('float32')
        img /= 255

        if fn[5].isdigit():
            label = int(fn[5])
        else:
            label = int(fn[4])
        # print(fn)
        # print(label)
        # print(np.argmax(model.predict(img.reshape(1, 28, 28, 1))[0]))
        x_train_more.append(img.reshape(28, 28, 1))
        y_train_more.append(label)

for i in xrange(30):
    re_train_acc = Model4(input_tensor=input_tensor, train=True, re_train=True, x_train_more=x_train_more,
                              y_train_more=y_train_more, is_retrain=1, epoch=i+1)
    print(re_train_acc)