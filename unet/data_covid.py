import os
import numpy as np
from keras import backend as K

K.set_image_data_format('channels_last')  # TF dimension ordering in this code
smooth = 1.

# The functions return our metric and loss
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def load_train_data():
    imgs_train_npz = np.load('/data/flavio/anatiel/datasets/A/train.npz')
    masks_train_npz = np.load('/data/flavio/anatiel/datasets/B_lesion/train.npz')
    imgs_train = imgs_train_npz['arr_0']
    masks_train = masks_train_npz['arr_0']
    
    return imgs_train, masks_train

def load_test_data():
    imgs_test_npz = np.load('/data/flavio/anatiel/datasets/A/test.npz')
    masks_test_npz = np.load('/data/flavio/anatiel/datasets/B_lesion/test.npz')
    imgs_test = imgs_test_npz['arr_0']
    masks_test = masks_test_npz['arr_0']

    return imgs_test, masks_test

if __name__ == '__main__':
    load_train_data()
    load_test_data()