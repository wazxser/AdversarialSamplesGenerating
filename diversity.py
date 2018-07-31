from scipy.misc import imsave, imread
from keras import backend as K
import numpy as np
import os

# dir = "./results/light_100/"
dir = "../../deepxplore/MNIST/generated_inputs_300/"
sum = 0
org = "orig"
count = 0
img1 = []
img2 = []
for fn in os.listdir(dir):
    if fn[0] == 'l':
        if not org in fn:
            img1.append(imread(dir+fn))
        else:
            img2.append(imread(dir+fn))
print(len(img1))
print(len(img2))

for i in xrange(len(img1)):
    sum += np.mean(abs(img1[i] - img2[i]))

print((sum/len(img1)).astype('float32'))