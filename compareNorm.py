import numpy as np
from scipy.misc import imsave, imread

# from keras.datasets import mnist
#
# (x_train, _), (_, _) = mnist.load_data()
#
# res = abs(np.linalg.norm(x_train[0], ord=np.inf) - np.linalg.norm(x_train[1], ord=np.inf))
# for i in xrange(101):
#     for j in xrange(101):
#         if i != j:
#             n1 = np.linalg.norm(x_train[i], ord=np.inf)
#             n2 = np.linalg.norm(x_train[j], ord=np.inf)
#
#             temp = abs(n1-n2)
#             if temp == 0:
#                 print(temp)
#             res = min(temp, res)
#
# print(res)
img1 = imread("./save.png").astype('float32') / 255
img2 = imread("./orig.png").astype('float32') / 255

# img1 = imread("./results/light/light_8_0.png").astype('float32') / 255
# img2 = imread("./results/light/light_8_0_orig.png").astype('float32') / 255

img = img1 - img2

print(np.linalg.norm(img, ord=np.inf))