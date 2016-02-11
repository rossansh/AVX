## import libs

import numpy as np
import matplotlib.pyplot as plt

from cs231nlib.classifier import NearestNeighbor
from cs231nlib.utils import load_CIFAR10
from cs231nlib.utils import visualize_CIFAR


## load dataset

Xtr, Ytr, Xte, Yte = load_CIFAR10('data/cifar10/') # a magic function we provide (stanford joke !)
# Xtr, Ytr: training data and labels
# Xte, Yte: testing data and labels


print Xtr.shape
# Xtr.shape[0] : 50000 (training images)
# Xtr.shape[1] & Xtr.shape[2] : 32x32 colour images
print Xte.shape
# Xte.shape[0] : 10000 (testing images)



# plt.rcParams['figure.figsize']=(10.0, 8.0)
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'

# visualize_CIFAR(X_train=Xtr, y_train=Ytr, samples_per_class=10)

# nn=NearestNeighbor();
