import cv2
import numpy as np

from keras.datasets import cifar10
from keras.utils import np_utils

nb_train_samples = 3000 
nb_valid_samples = 100 
num_classes = 10

def load_cifar10_data(img_rows, img_cols):

    # 加载数据
    (X_train, Y_train), (X_valid, Y_valid) = cifar10.load_data()

    # 统一大小
    X_train = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_train[:nb_train_samples,:,:,:]])
    X_valid = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_valid[:nb_valid_samples,:,:,:]])

    # 转换成所需格式
    Y_train = np_utils.to_categorical(Y_train[:nb_train_samples], num_classes)
    Y_valid = np_utils.to_categorical(Y_valid[:nb_valid_samples], num_classes)

    return X_train, Y_train, X_valid, Y_valid