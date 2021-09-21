import tensorflow as tf
# print(tf.__version__)
# tf.config.run_functions_eagerly(True)

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
    bce_ = bce(y_true, y_pred).numpy()
    
    dice_bce_loss = (dice_loss + bce_) / 2
    
    return dice_bce_loss

def load_train_data():
    # remote
    imgs_train_npz = np.load('/home/anatiel/datasets/dissertacao/train_images_ds1_32bits.npz', allow_pickle=True)
    masks_train_npz = np.load('/home/anatiel/datasets/dissertacao/dissertacao/train_masks_ds1_32bits.npz', allow_pickle=True)

    # local
    # imgs_train_npz = np.load('/home/anatielsantos/mestrado/datasets/dissertacao/train_images_ds1_equalize_hist.npz', allow_pickle=True)
    # masks_train_npz = np.load('/home/anatielsantos/mestrado/datasets/dissertacao/train_masks_ds1_equalize_hist.npz', allow_pickle=True)

    imgs_train = imgs_train_npz['arr_0']
    masks_train = masks_train_npz['arr_0']
    
    return np.expand_dims(np.concatenate(imgs_train), axis=-1), np.expand_dims(np.concatenate(masks_train), axis=-1)

def load_val_data():
    imgs_val_npz = np.load('/data/flavio/anatiel/datasets/A/512x512/val_masked_lung.npz')
    masks_val_npz = np.load('/data/flavio/anatiel/datasets/B_lesion/512x512/val_masked.npz')
    imgs_val = imgs_val_npz['arr_0']
    masks_val = masks_val_npz['arr_0']
    
    return imgs_val, masks_val

def load_test_data():
    imgs_test_npz = np.load('/data/flavio/anatiel/datasets/A/512x512/test_lung.npz')
    masks_test_npz = np.load('/data/flavio/anatiel/datasets/B_lesion/512x512/test.npz')
    imgs_test = imgs_test_npz['arr_0']
    masks_test = masks_test_npz['arr_0']

    return imgs_test, masks_test

# results train
def results_train(history):
    result = np.mean(history)
    
    return result

if __name__ == '__main__':
    load_train_data()
    # load_test_data()