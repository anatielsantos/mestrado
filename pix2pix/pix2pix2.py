# GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

#configuração necessária nas GPU's RTX 
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

from tensorflow.keras.callbacks import ModelCheckpoint
from model import Pix2Pix
from utils import *
from losses import *

import numpy as np
import pandas as pd
import skimage.io as io
import matplotlib.pyplot as plt

# train settings
BUFFER_SIZE = 400
BATCH_SIZE = 1
EPOCHS = 100
IMG_WIDTH = 512
IMG_HEIGHT = 512
INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 1
path_weights = '/data/flavio/anatiel/models/'

# load dataset
def load_images(path_src, path_mask):
    src = np.expand_dims(np.load(path_src)['arr_0'], axis=-1)
    tar = np.expand_dims(np.load(path_mask)['arr_0'].astype(np.float32), axis=-1)
    
    return [src,tar]

# dataset path
path_src_train = '/data/flavio/anatiel/datasets/A/512x512/train.npz'
path_mask_train = '/data/flavio/anatiel/datasets/B_lesion/512x512/train.npz'
path_src_val = '/data/flavio/anatiel/datasets/A/512x512/val.npz'
path_mask_val = '/data/flavio/anatiel/datasets/B_lesion/512x512/val.npz'

# load dataset
[src_images_train, tar_images_train] = load_images(path_src_train, path_mask_train)
[src_images_val, tar_images_val] = load_images(path_src_val, path_mask_val)
print('Loaded train images: ', src_images_train.shape, tar_images_train.shape)
print('Loaded val images: ', src_images_val.shape, tar_images_val.shape)

def train(src_images_train, tar_images_train, src_images_val, tar_images_val):    
    dataset = [src_images_train, tar_images_train]

    # createing pix2pix
    model = Pix2Pix(IMG_HEIGHT,IMG_WIDTH,INPUT_CHANNELS,OUTPUT_CHANNELS)
    model.compile(
        discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
        generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
        discriminator_loss = discriminator_loss,
        generator_loss = generator_loss
    )

    # train model
    checkpoint = ModelCheckpoint(path_weights+'best_weights_train_gan_aumentation.hdf5', monitor='dice', verbose=1, save_best_only=True,save_weights_only=True, mode='max')
    checkpoint2 = ModelCheckpoint(path_weights+'best_weights_val_gan_aumentation.hdf5', monitor='val_dice', verbose=1, save_best_only=True,save_weights_only=True, mode='max')
    history=model.fit(src_images_train, tar_images_train, batch_size=BATCH_SIZE, epochs=EPOCHS,callbacks=[checkpoint,checkpoint2],validation_data=(src_images_val, tar_images_val))
    
    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history) 
    
    # save to json:  
    print("anatiel/tf_pix2pix/saving history")
    hist_json_file = 'anatiel/tf_pix2pix/history.json' 
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)
    print("history saved")
    
    plt.plot(history.history['dice'])
    plt.title('Model dice coeff')
    plt.ylabel('Dice coeff')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    # save plot to file
    plt.savefig('anatiel/tf_pix2pix/plot_dice.png')
    plt.show()

if __name__=="__main__":
    # model training
    train(src_images_train, tar_images_train, src_images_val, tar_images_val)