import keras.tensorflow as tf
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