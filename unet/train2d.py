from __future__ import print_function

# GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, History

from skimage.segmentation import mark_boundaries
from skimage.exposure import rescale_intensity
from skimage.transform import resize
from skimage.io import imsave
import skimage.transform as trans
import skimage.io as io
from data_covid import load_train_data, load_val_data, load_test_data, dice_coef, dice_coef_loss
from losses import *


BATCH_SIZE = 1
EPOCHS = 500

# model
def unet(pretrained_weights = None,input_size = (512,512,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(32, (3, 3), padding='same')(inputs)
    conv1 = LeakyReLU()(conv1)
    conv1 = Dropout(0.2)(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3),padding='same')(conv1)
    conv1 = LeakyReLU()(conv1)
    conv1 = BatchNormalization()(conv1)
    conc1 = concatenate([inputs, conv1], axis=3)
    pool1 = AveragePooling2D(pool_size=(2, 2))(conc1)


    conv2 = Conv2D(64, (3, 3), padding='same')(pool1)
    conv2 = LeakyReLU()(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), padding='same')(conv2)
    conv2 = LeakyReLU()(conv2)
    conv2 = BatchNormalization()(conv2)
    conc2 = concatenate([pool1, conv2], axis=3)
    pool2 = AveragePooling2D(pool_size=(2, 2))(conc2)


    conv3 = Conv2D(128, (3, 3), padding='same')(pool2)
    conv3 = LeakyReLU()(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), padding='same')(conv3)
    conv3 = LeakyReLU()(conv3)
    conv3 = BatchNormalization()(conv3)
    conc3 = concatenate([pool2, conv3], axis=3)
    pool3 = AveragePooling2D(pool_size=(2, 2))(conc3)


    conv4 = Conv2D(256, (3, 3), padding='same')(pool3)
    conv4 = LeakyReLU()(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(256, (3, 3), padding='same')(conv4)
    conv4 = LeakyReLU()(conv4)
    conv4 = BatchNormalization()(conv4)
    conc4 = concatenate([pool3, conv4], axis=3)
    pool4 = AveragePooling2D(pool_size=(2, 2))(conc4)


    conv5 = Conv2D(512, (3, 3), padding='same')(pool4)
    conv5 = LeakyReLU()(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(512, (3, 3), padding='same')(conv5)
    conv5 = LeakyReLU()(conv5)
    conv5 = BatchNormalization()(conv5)
    conc5 = concatenate([pool4, conv5], axis=3)


    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conc5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), padding='same')(up6)
    conv6 = LeakyReLU()(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(256, (3, 3), padding='same')(conv6)
    conv6 = LeakyReLU()(conv6)
    conv6 = BatchNormalization()(conv6)
    conc6 = concatenate([up6, conv6], axis=3)


    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conc6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), padding='same')(up7)
    conv7 = LeakyReLU()(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(128, (3, 3), padding='same')(conv7)
    conv7 = LeakyReLU()(conv7)
    conv7 = BatchNormalization()(conv7)
    conc7 = concatenate([up7, conv7], axis=3)


    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conc7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), padding='same')(up8)
    conv8 = LeakyReLU()(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Conv2D(64, (3, 3), padding='same')(conv8)
    conv8 = LeakyReLU()(conv8)
    conv8 = BatchNormalization()(conv8)
    conc8 = concatenate([up8, conv8], axis=3)


    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conc8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), padding='same')(up9)
    conv9 = LeakyReLU()(conv9)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Conv2D(32, (3, 3), padding='same')(conv9)
    conv9 = LeakyReLU()(conv9)
    conc9 = concatenate([up9, conv9], axis=3)


    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conc9)

    model = Model(inputs=inputs, outputs=[conv10])

    # model.compile(optimizer=Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199), loss=dice_coef_loss, metrics=['accuracy', dice_coef])

    otimizador = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199)
    model.compile(optimizer=otimizador, loss=dice_coef_loss, metrics=['accuracy',dice_coef])
    # self.model.compile(optimizer=otimizador, loss=focal_tversky, metrics=['accuracy',dice_jonnison])
    # exit()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

# train
def train():
    print('Loading and preprocessing train data...')
    print('-'*30)

    imgs_train, imgs_mask_train = load_train_data()
    print('Loaded train images: ', imgs_train.shape, imgs_mask_train.shape)
    print('-'*30)
    
    imgs_val, imgs_mask_val = load_val_data()
    print('Loaded val images: ', imgs_val.shape, imgs_mask_val.shape)
    print('-'*30)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    #Normalization of the train set
    imgs_train -= mean
    imgs_train /= std
    imgs_mask_train = imgs_mask_train.astype('float32')

    print('Creating and compiling model...')
    print('-'*30)
    
    model = unet()
    #Saving the weights and the loss of the best predictions we obtained
    model_checkpoint = ModelCheckpoint('/data/flavio/anatiel/models/unet2d/weights_unet_masked_lung_clahe_500epc.h5', monitor='val_loss', save_best_only=True)
    # checkpoint = ModelCheckpoint('/data/flavio/anatiel/models/unet2d/weights_train_unet_masked_lung_blur_3epc.h5', monitor='dice', verbose=1, save_best_only=True,save_weights_only=True, mode='max')
    # checkpoint2 = ModelCheckpoint('/data/flavio/anatiel/models/unet2d/weights_val_unet_masked_lung_blur_3epc.h5', monitor='val_dice', verbose=1, save_best_only=True,save_weights_only=True, mode='max')
    
    print('Fitting model...')
    print('-'*30)
    history = model.fit(imgs_train, imgs_mask_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, shuffle=True, validation_split=0.2, callbacks=[model_checkpoint])
    # history = model.fit(imgs_train, imgs_mask_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, shuffle=True, callbacks=[checkpoint,checkpoint2], validation_data=(imgs_val, imgs_mask_val))

    model.save('/data/flavio/anatiel/models/unet2d/weights_unet_masked_lung_clahe_500epc_last.h5')
        
    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history) 
    
    # save to json:  
    print("saving history...")
    hist_json_file = 'anatiel/unet/history_masked_lung_clahe_500epc.json' 
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)
    print("history saved")
    
    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title('Model dice coeff')
    plt.ylabel('Dice coeff')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    # save plot to file
    plt.savefig('anatiel/unet/plot_dice_masked_lung_clahe_500epc.png')
    plt.show()
    
if __name__ == "__main__":
    # model training
    train()