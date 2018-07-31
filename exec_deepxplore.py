import os

arr = [0, 1, 2]
for i in arr:
    # os.system('sudo python deepxplore_mnist.py light 1 0 1 500 50 0 ' + str(i))
    # os.system('sudo python deepxplore_mnist.py light 1 0.1 1 500 50 0 ' + str(i))
    # os.system('sudo python deepxplore_mnist.py light 1 0.3 1 500 50 0 ' + str(i))
    os.system('sudo python deepxplore_mnist.py light 1 1 1 500 50 0 ' + str(i))

