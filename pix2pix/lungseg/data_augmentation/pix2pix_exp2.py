# GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

#configuração necessária nas GPU's RTX 
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
# session = tf.compat.v1.InteractiveSession(config=config)
# session.close()

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import Pix2Pix
from losses import *

import numpy as np
import pandas as pd
import skimage.io as io
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# train settings
BUFFER_SIZE = 400
BATCH_SIZE = 1
EPOCHS = 100
IMG_WIDTH = 640
IMG_HEIGHT = 640
INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 1

# loss
def generator_loss(disc_generated_output, gen_output, target,LAMBDA=100):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    #l1_loss = tf.reduce_mean(tf.abs(target - gen_output)) # default
    
    # one loss function
    # l1_loss = dice_loss(target, gen_output)

    # two loss function
    l1_loss = dice_bce_loss(target, gen_output)

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss

# load dataset
def load_images(path_src, path_mask):
    # src = np.expand_dims(np.load(path_src)['arr_0'], axis=-1)
    # tar = np.expand_dims(np.load(path_mask)['arr_0'].astype(np.float32), axis=-1)
        
    # return [src,tar]

    src_npz = np.load(path_src, allow_pickle=True)
    tar_npz = np.load(path_mask, allow_pickle=True)
    src = src_npz['arr_0']
    tar = tar_npz['arr_0']
    
    return np.float32(np.expand_dims(np.concatenate(src), axis=-1)), np.float32(np.expand_dims(np.concatenate(tar), axis=-1))

def train(path_weights, src_images_train, tar_images_train):    
    # dataset = [src_images_train, tar_images_train]

    # createing pix2pix
    model = Pix2Pix(IMG_HEIGHT,IMG_WIDTH,INPUT_CHANNELS,OUTPUT_CHANNELS)
    model.compile(
        discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
        generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
        discriminator_loss = discriminator_loss,
        generator_loss = generator_loss,
        metrics=['accuracy', dice]
    )

    # Normalization of the train set #1 and #2
    mean = np.mean(src_images_train)  # mean for data centering
    std = np.std(src_images_train)  # std for data normalization
    src_images_train -= mean
    src_images_train /= std

    print('Train test split')
    X_train, X_test, y_train, y_test = train_test_split(src_images_train, tar_images_train, test_size=0.1)

    print('-'*30)
    print('Data Augmentation Start')
    data_gen_args = dict(shear_range=0.1,
			rotation_range=20,
			width_shift_range=0.1, 
			height_shift_range=0.1,
			zoom_range=0.3,
			fill_mode='constant',
			horizontal_flip=True,
			vertical_flip=True,
			cval=0)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    seed = 1
    image_datagen.fit(X_train, augment=True, seed=seed)
    mask_datagen.fit(y_train, augment=True, seed=seed)

    image_generator = image_datagen.flow(X_train, batch_size = BATCH_SIZE)
    mask_generator = mask_datagen.flow(y_train, batch_size = BATCH_SIZE)

    train = zip(image_generator, mask_generator)
    # val = zip(X_test, y_test)

    print('-'*30)
    print('Data Augmentation End')
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)

    # train model
    checkpoint = ModelCheckpoint(path_weights+'gan_lungseg_exp2_100epc_augment_best.hdf5', monitor='dice', verbose=1, save_best_only=True,save_weights_only=True, mode='max')
    #checkpoint2 = ModelCheckpoint(path_weights+'best_weights_val_gan_512_masked_lung_blur_500epc.hdf5', monitor='val_dice', verbose=1, save_best_only=True,save_weights_only=True, mode='max')
    
    # history = model.fit(src_images_train, tar_images_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, shuffle=True, validation_split=0.1, callbacks=[checkpoint])
    history=model.fit(train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, shuffle=True,  validation_data=(X_test, y_test), callbacks=[checkpoint])
    
    model.save(path_weights+'gan_lungseg_exp2_100epc_augment_last.hdf5')
    
    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history) 
    
    # save to json:  
    print("Saving history")
    hist_json_file = path_json+'gan_lungseg_exp2_100epc_augment_history.json' 
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)
    print("History saved")
    
    plt.plot(history.history['dice'])
    plt.plot(history.history['val_dice'])
    plt.plot(history.history['g_l1'])
    plt.title('Model dice coeff')
    plt.ylabel('Dice coeff')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val', 'Loss'], loc='upper left')
    # save plot to file
    plt.savefig(path_plot+'gan_lungseg_exp2_100epc_augment_plot.png')
    # plt.show()

if __name__=="__main__":
    # dataset path remote
    path_src_train = '/data/flavio/anatiel/datasets/dissertacao/train_images_ds1_32bits.npz'
    path_mask_train = '/data/flavio/anatiel/datasets/dissertacao/train_masks_ds1_32bits.npz'

    # dataset path local
    # path_src_train = '/home/anatielsantos/mestrado/datasets/dissertacao/train_images.npz'
    # path_mask_train = '/home/anatielsantos/mestrado/datasets/dissertacao/train_masks.npz'

    # paths save
    path_weights = '/data/flavio/anatiel/models/dissertacao/'
    path_json = '/data/flavio/anatiel/models/dissertacao/'
    path_plot = '/data/flavio/anatiel/models/dissertacao/'

    # load dataset
    [src_images_train, tar_images_train] = load_images(path_src_train, path_mask_train)
    print('Loaded train images: ', src_images_train.shape, tar_images_train.shape)

    # model training
    train(path_weights, src_images_train, tar_images_train)
