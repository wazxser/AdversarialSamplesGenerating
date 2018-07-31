from Model1 import Model1
from Model2 import Model2
from Model3 import Model3
from utils import *
from keras.layers import Input
from keras.datasets import mnist
import sys


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
model = Model2(input_tensor=input_tensor)

layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name]
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
model_layer_dict = init_coverage_tables(model)
all_neuron_num = neuron_covered(model_layer_dict)[1]
high_out = [-sys.maxint-1] * all_neuron_num
low_out = [sys.maxint] * all_neuron_num
k = 1000
for num in xrange(60000):
    input_data = x_train[num]
    input_data = np.expand_dims(input_data, axis=0)
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    index = 0
    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        # scaled = scale(intermediate_layer_output[0])
        for num_neuron in xrange(intermediate_layer_output[0].shape[-1]):
            neuron_out = np.mean(intermediate_layer_output[0][..., num_neuron])
            if neuron_out < low_out[index]:
                low_out[index] = neuron_out
            if neuron_out > high_out[index]:
                high_out[index] = neuron_out
            index += 1

k_cover = [[0] * k] * all_neuron_num
upper_cover = [0] * all_neuron_num
lower_cover = [0] * all_neuron_num

for num2 in xrange(10000):
    input_data = x_test[num2]
    input_data = np.expand_dims(input_data, axis=0)
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    index = 0
    for j, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        # scaled = scale(intermediate_layer_output[0])
        for num_neuron in xrange(intermediate_layer_output[0].shape[-1]):
            neuron_out = np.mean(intermediate_layer_output[0][..., num_neuron])

            if neuron_out > high_out[index]:
                upper_cover[index] = 1
            elif neuron_out < low_out[index]:
                lower_cover[index] = 1
            else:
                if high_out[index] - low_out[index] == 0:
                    out_sec = 0
                else:
                    out_sec = ((neuron_out - low_out[index]) / ((high_out[index] - low_out[index]) / k)).astype('int')
                if out_sec == k:
                    k_cover[index][k-1] = 1
                else:
                    k_cover[index][out_sec] = 1
            index += 1

print(sum(map(sum, k_cover)))
print(sum(map(sum, k_cover)) * 1.0 / (k * all_neuron_num))
print(sum(upper_cover) * 1.0 / all_neuron_num)
print(sum(lower_cover) * 1.0 / all_neuron_num)

print(sum(upper_cover))
print(sum(lower_cover))
