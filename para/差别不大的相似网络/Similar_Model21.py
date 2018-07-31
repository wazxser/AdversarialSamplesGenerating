'''
LeNet-4
'''

# usage: python MNISTSimilar_Model21.py - train the model

from __future__ import print_function

from keras.datasets import mnist
from keras.layers import Convolution2D, MaxPooling2D, Input, Dense, Activation, Flatten
from keras.models import Model
from keras.utils import to_categorical

from configs import bcolors
import numpy as np

def Similar_Model21(input_tensor=None, train=False, re_train=False, x_train_more=[], y_train_more=[]):
    nb_classes = 10
    # convolution kernel size
    kernel_size = (5, 5)
    nb_epoch = 2

    if train:
        batch_size = 128

        # input image dimensions
        img_rows, img_cols = 28, 28

        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        # if re_train:
        #     x_train = np.append(x_train, x_train_more, axis=0)
        #     y_train = np.append(y_train, y_train_more, axis=0)

        # convert class vectors to binary class matrices
        y_train = to_categorical(y_train, nb_classes)
        y_test = to_categorical(y_test, nb_classes)

        input_tensor = Input(shape=input_shape)
    elif input_tensor is None:
        print(bcolors.FAIL + 'you have to proved input_tensor when testing')
        exit()

    x = Dense(100, activation='relu', name='fc1')(input_tensor)
    x = Dense(100, activation='relu', name='fc2')(x)
    x = Dense(100, activation='relu', name='fc3')(x)
    x = Dense(100, activation='relu', name='fc4')(x)
    x = Dense(80, activation='relu', name='fc5')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(nb_classes, name='before_softmax')(x)
    x = Activation('softmax', name='predictions')(x)

    model = Model(input_tensor, x)

    if train:
        # compiling
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        if re_train:
            model.load_weights('./Similar_Model21.h5')
        # trainig
        model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=nb_epoch, verbose=1)
        # save model
        if re_train:
            model.save_weights('./Similar_Model21.h5')
            # model.save_weights('./Similar_Model21_retrain_' + str(nb_epoch) + '_robustness.h5')
        else:
            model.save_weights('./Similar_Model21.h5')
        score = model.evaluate(x_test, y_test, verbose=0)
        print('\n')
        print('Overall Test score:', score[0])
        print('Overall Test accuracy:', score[1])
        return score[1]
    else:
        # model.load_weights('./Similar_Model21_retrain_' + str(nb_epoch) + '_robustness.h5')
        model.load_weights('./Similar_Model21.h5')
        print(bcolors.OKBLUE + 'Similar_Model21 loaded' + bcolors.ENDC)

    return model


if __name__ == '__main__':
    Similar_Model21(train=True, re_train=True)
