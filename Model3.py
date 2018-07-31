'''
LeNet-5
'''

# usage: python MNISTModel3.py - train the model

from __future__ import print_function

from keras.datasets import mnist
from keras.layers import Convolution2D, MaxPooling2D, Input, Dense, Activation, Flatten
from keras.models import Model
from keras.utils import to_categorical

from configs import bcolors
import numpy as np
from matplotlib import pyplot


def plot_loss_curve(model):
    pyplot.plot(model.history.history['loss'])
    pyplot.plot(model.history.history['val_loss'])
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.show()


def plot_acc_curve(model):
    pyplot.plot(model.history.history['acc'])
    pyplot.plot(model.history.history['val_acc'])
    pyplot.ylabel('acc')
    pyplot.xlabel('epoch')
    pyplot.show()


def Model3(input_tensor=None, train=False, re_train=False, x_train_more=[], y_train_more=[], retrain_num=0):
    nb_classes = 10
    kernel_size = (5, 5)
    nb_epoch = 300

    if train:
        batch_size = 256

        img_rows, img_cols = 28, 28

        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        if re_train:
            x_train = np.append(x_train, x_train_more, axis=0)
            y_train = np.append(y_train, y_train_more, axis=0)

        y_train = to_categorical(y_train, nb_classes)
        y_test = to_categorical(y_test, nb_classes)

        input_tensor = Input(shape=input_shape)
    elif input_tensor is None:
        print(bcolors.FAIL + 'you have to proved input_tensor when testing')
        exit()

    x = Convolution2D(6, kernel_size, activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)

    x = Convolution2D(16, kernel_size, activation='relu', padding='same', name='block2_conv1')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block2_pool1')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(120, activation='relu', name='fc1')(x)
    x = Dense(84, activation='relu', name='fc2')(x)
    x = Dense(nb_classes, name='before_softmax')(x)
    x = Activation('softmax', name='predictions')(x)

    model = Model(input_tensor, x)

    if train:
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

        # if re_train:
        #     model.load_weights('./Model3.h5')

        model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=nb_epoch, verbose=1)

        # if re_train:
        #     model.save_weights('./Model3_retrain_' + str(nb_epoch) + '_robustness.h5')
        # else:
        #     model.save_weights('./Model3.h5')

        model.save_weights('./Model3' + '_' + str(nb_epoch) + '.h5')
        score = model.evaluate(x_test, y_test, verbose=0)
        print('\n')
        print('Overall Test score:', score[0])
        print('Overall Test accuracy:', score[1])
        plot_loss_curve(model)
        plot_acc_curve(model)
        return score[1]
    else:
        model.load_weights('./Model3' + '_' + str(retrain_num) + '.h5')
        print(bcolors.OKBLUE + 'Model3' + '_' + str(retrain_num) + ' loaded' + bcolors.ENDC)

    return model


if __name__ == '__main__':
    Model3(train=True)
