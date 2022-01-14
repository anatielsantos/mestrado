import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import backend as K

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

# média do dice_loss + BCE
def dice_bce_loss(y_true, y_pred):
    dice_loss = -dice_coef(y_true, y_pred)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    bce_ = bce(y_true, y_pred)#.numpy()

    dice_bce_loss = (dice_loss + bce_) / 2
    
    return dice_bce_loss


def load_train_data():
    # remote
    imgs_train_npz = np.load('/data/flavio/anatiel/datasets/dissertacao/images_fold_1.npz', allow_pickle=True)
    masks_train_npz = np.load('/data/flavio/anatiel/datasets/dissertacao/masks_fold_1.npz', allow_pickle=True)

    imgs_train = imgs_train_npz['arr_0']
    masks_train = masks_train_npz['arr_0']

    return np.expand_dims(np.concatenate(imgs_train), axis=-1), np.expand_dims(np.concatenate(masks_train), axis=-1)
    

if __name__ == '__main__':
    load_train_data()
