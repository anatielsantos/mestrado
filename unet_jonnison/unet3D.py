from __future__ import print_function
# from utils.nvidia import set_keras_backend
# set_keras_backend('tensorflow')

import os
import random,json
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv3D, MaxPooling3D, Conv3DTranspose, AveragePooling3D, ZeroPadding3D
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
import teste_load

from losses import *

class Unet3D():
    def __init__(self,img_depth,img_rows,img_cols,img_channels=1,N_CLASSES=2,batchsize=1,path_weights=None):
        self.img_rows=img_rows
        self.img_cols=img_cols
        self.img_depth=img_depth
        self.img_channels=img_channels
        self.N_CLASSES=N_CLASSES
        self.batchsize=batchsize
        self.path_weights=path_weights
        self.build_model()

        # Configure data loader
        #self.dataset_name = 'facades'
        #self.data_loader = DataLoader_(dataset_name=self.dataset_name, img_res=(self.img_rows, self.img_cols))
        #self.data_loader = DataLoader_()
        
    def build_model(self):

        inputs = Input((self.img_depth,self.img_rows, self.img_cols, self.img_channels))
        conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

        conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

        conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

        conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

        conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv5)

        up6 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv5), conv4], axis=4)
        conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv6)

        up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6), conv3], axis=4)
        conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv7)

        up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7), conv2], axis=4)
        conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv8)

        up9 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv8), conv1], axis=4)
        conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv9)

        conv10 = Conv3D(self.N_CLASSES, (1, 1, 1), activation='sigmoid')(conv9)


        self.model = Model(inputs=[inputs], outputs=[conv10])
        self.model.compile(optimizer=Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199), loss=dice_loss, metrics=['accuracy',dice])

    def load_weights(self,filename):
        self.model.load_weights(self.path_weights+filename)
    
    def save_weights(self,filename):
        self.model.save_weights(self.path_weights+filename)


    def train(self,X_train,y_train,X_val,y_val,epochs):
        checkpoint = ModelCheckpoint(str(self.path_weights)+'best_weights_train_unet.hdf5', monitor='dice', verbose=1, save_best_only=True,save_weights_only=True, mode='max')
        checkpoint2 = ModelCheckpoint(str(self.path_weights)+'best_weights_val_unet.hdf5', monitor='val_dice', verbose=1, save_best_only=True,save_weights_only=True, mode='max')
        history = self.model.fit(X_train, y_train, 
                  epochs=epochs,
                  batch_size=self.batchsize,
                  callbacks=[checkpoint,checkpoint2],
                  validation_data= (X_val,y_val))
        self.save_weights('end_weights_unet.hdf5')
        # print(history.history)
        for key in history.history.keys():
            history.history[key] = [float(x) for x in history.history[key]]
        file = open(self.path_weights+"history.json",'w')
        file.write(json.dumps(history.history))
        file.close()

    def train_generator(self,train_generator,train_steps,val_generator,val_steps,epochs):
        checkpoint = ModelCheckpoint(self.path_weights+'best_weights_train_unet.hdf5', monitor='dice', verbose=1, save_best_only=True,save_weights_only=True, mode='max')
        checkpoint2 = ModelCheckpoint(self.path_weights+'best_weights_val_unet.hdf5', monitor='val_dice', verbose=1, save_best_only=True,save_weights_only=True, mode='max')
        history = self.model.fit(train_generator, 
                  steps_per_epoch = train_steps,
                  validation_data= val_generator,
                  validation_steps = val_steps,
                  callbacks=[checkpoint,checkpoint2],
                  epochs=epochs)
        self.save_weights('end_weights_unet.hdf5')

        file = open(self.path_weights+"history.json",'w')
        file.write(json.dumps(history.history))
        file.close()

    def predict(self,X_test):
        pred_test = self.model.predict(X_test,batch_size=self.batchsize)
        #pred_test = (pred_test>0.5)*1
        return pred_test

if __name__ == '__main__':
    print("Carregando imagens")
    X_train, y_train = teste_load.load_data_train()
    X_val, y_val = teste_load.load_data_val()

    epochs = 5
    img_depth = 272
    img_rows = 304
    img_cols = 432

    print("Iniciando treino")
    model = Unet3D(img_depth, img_rows, img_cols)
    model.train(np.array(X_train),np.array(y_train),np.array(X_val),np.array(y_val),epochs)