from __future__ import print_function

# GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

from keras.models import Model
#from keras.models import *

#from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import *

from keras.optimizers import Adam, SGD
#from keras.optimizers import *

from keras import backend as K
#from keras import backend as keras

from skimage.exposure import rescale_intensity
import skimage.transform as trans

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, History

import skimage.io as io
import cv2
from data_covid import load_train_data, load_test_data

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = int(512/2)
img_cols = int(512/2)
smooth = 1.
rede = 'unet/'
pre_set = ''
path_weights = '/data/flavio/anatiel/models/new/'+rede
path_preds = '/data/flavio/anatiel/preds/'+rede+'dice_bce'
#We divide here the number of rows and columns by two because we undersample our data (We take one pixel over two)

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

# IoU
def iou(y_true, y_pred):
    '''intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score'''
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    iou_ = intersection/union
    return iou_

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

def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p

#We adapt here our dataset samples dimension so that we can feed it to our network

def morphological_operations(im_mask_gan, op): 
    kernel = np.ones((5,5),np.uint8)

    if op == 'none':
        operation = np.float32(im_mask_gan)
    if op == 'median':
        operation = cv2.medianBlur(np.float32(im_mask_gan), 5)
    if op == 'erode':
        operation = cv2.erode(np.float32(im_mask_gan), kernel, iterations=1)
    if op == 'dilate':
        operation = cv2.dilate(np.float32(im_mask_gan), kernel, iterations=1)
    if op == 'opening':
        operation = cv2.morphologyEx(np.float32(im_mask_gan), cv2.MORPH_OPEN, kernel)
    if op == 'closing':
        operation = cv2.morphologyEx(np.float32(im_mask_gan), cv2.MORPH_CLOSE, kernel)
    
    return operation

# def train():
#     print('-'*30)
#     print('Loading and preprocessing train data...')
#     print('-'*30)
#     imgs_train, imgs_mask_train = load_train_data()
#     print('Loaded train images: ', imgs_train.shape, imgs_mask_train.shape)

#     #imgs_train = preprocess(imgs_train)
#     #imgs_mask_train = preprocess(imgs_mask_train)

#     imgs_train = imgs_train.astype('float32')
#     mean = np.mean(imgs_train)  # mean for data centering
#     std = np.std(imgs_train)  # std for data normalization

#     imgs_train -= mean
#     imgs_train /= std
#     #Normalization of the train set

#     imgs_mask_train = imgs_mask_train.astype('float32')

#     print('-'*30)
#     print('Creating and compiling model...')
#     print('-'*30)
#     #model = get_unet()
#     model = unet()
#     model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)
#     #Saving the weights and the loss of the best predictions we obtained

#     print('-'*30)
#     print('Fitting model...')
#     print('-'*30)
#     history=model.fit(imgs_train, imgs_mask_train, batch_size=1, epochs=50, verbose=1, shuffle=True,
#               validation_split=0.2,
#               callbacks=[model_checkpoint])

def test(op):
    print('-'*30)
    print('Loading and preprocessing test data...')

    imgs_test, imgs_maskt = load_test_data()
    
    #imgs_test = imgs_test[:16]
    #imgs_maskt = imgs_maskt[:16]
    
    imgs_test = imgs_test.astype('float32')
    mean = np.mean(imgs_test)  # mean for data centering
    std = np.std(imgs_test)  # std for data normalization
    
    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std
    
    imgs_maskt = imgs_maskt.astype('float32')
    #Normalization of the test set

    print('-'*30)
    print('Loading saved weights...')
    
    model = unet()
    model.load_weights('/data/flavio/anatiel/models/dice_bce_weights_unet_masked_lung'+pre_set+'_500epc_last.h5')

    print('-'*30)
    print('Predicting masks on test data...')
    
    imgs_mask_test = model.predict(imgs_test, batch_size=1)
    saida = imgs_mask_test
    
    for i in range(imgs_mask_test.shape[0]):
        try:
            saida[i,:,:] = morphological_operations(imgs_mask_test[i], op)
        except:
            saida[i,:,:,0] = morphological_operations(imgs_mask_test[i], op)
        
    np.save(path_preds+'imgs_mask_test'+pre_set+'_'+op+'_last.npy', saida)
    
    #mask_pred = np.load('imgs_mask_test.npy')
    
    print('-' * 30)
    print('Saving predicted masks to files...')
    #print('-' * 30)
    
    #dice_test = dice_coef(imgs_mask, mask_pred)
    #print("DICE Test: ", dice_test.numpy())
    
    #iou_test = iou(imgs_maskt, mask_pred)
    #print("IoU Test: ", iou_test.numpy())

def show_preds(preds, fatia):
    # load array
    data = np.load(path_preds+preds)
    
    print('-' * 30)
    print('Showing predicted masks (slice: ', fatia, ') ...')
    print('-' * 30)

    imgs_test, imgs_mask_test = load_test_data()
    im = imgs_test[fatia,:,:]
    im_mask = imgs_mask_test[fatia]
    
    print(np.amax(im))
    print(np.amin(im))
    
    dice_test = dice_coef(im_mask.astype('float32'), data[fatia,:,:,0])
    print("DICE Test: ", dice_test.numpy())
    
#     iou_test = iou(im_mask.astype('float32'), data[fatia,:,:,0])
#     print("IoU Test: ", iou_test.numpy())

    
    fig, ax = plt.subplots(ncols=3, figsize=(12, 5))
    ax[0] = plt.subplot(1, 3, 1)
    ax[1] = plt.subplot(1, 3, 2)
    ax[2] = plt.subplot(1, 3, 3)

    ax[0].imshow(im, cmap='gray')
    ax[0].set_title('CT')

    ax[1].imshow(im_mask, cmap="gray")
    ax[1].set_title('Máscara Médico')

    ax[2].imshow(data[fatia,:,:,0], cmap="gray")
    ax[2].set_title('Máscara GAN')

    plt.show()

if __name__ == "__main__":
    # model training
    #train()
    
    # predict
    test('none')
    test('median')
    test('erode')
    test('dilate')
    test('opening')
    test('closing')

# show
show_preds('imgs_mask_test_none.npy', 78)
show_preds('imgs_mask_test_median.npy', 78)
show_preds('imgs_mask_test_erode.npy', 78)
show_preds('imgs_mask_test_dilate.npy', 78)
show_preds('imgs_mask_test_opening.npy', 78)
show_preds('imgs_mask_test_closing.npy', 78)

"""===============New Tests - UNet Denes================
### 500 epochs (Masked Lung)
#### Train Bast
DICE: 0.9026
#### Test Bast
DICE:  0.6977257104312148
IoU: 0.5357747718894526
Sensitivity:  0.9402988527017024
Specificity 0.9977292515844518
ACC:  0.9975570470636541
AUC:  0.9690140521430771
Prec:  0.5546421656845262
FScore:  0.6977257104312148
#### Train Last
DICE: 0.9406
#### Test Last
DICE:  0.67896206554329
IoU: 0.513961066396265
Sensitivity:  0.940858850628053
Specificity 0.9976096275307739
ACC:  0.9974467745694248
AUC:  0.9692342390794134
Prec:  0.5311199130058682
FScore:  0.67896206554329

### 500 epochs (Masked Lung Clahe)
#### Train Bast
DICE: 0.8907
#### Test Bast
DICE:  0.602493504295082
IoU: 0.43112036054700226
Sensitivity:  0.6475318304009736
Specificity 0.997770277079672
ACC:  0.9962214313853871
AUC:  0.8226510537403229
Prec:  0.5633129264602148
FScore:  0.602493504295082
#### Train Last
DICE: 0.9258
#### Test Last
DICE:  0.5796817538951077
IoU: 0.4081351172420955
Sensitivity:  0.6633302506795318
Specificity 0.9975235920545211
ACC:  0.996205229325728
AUC:  0.8304269213670264
Prec:  0.5147675813148175
FScore:  0.5796817538951077

### 500 epochs (Masked Lung Blur)
#### Train Bast
DICE: 0.8834
#### Test Bast
DICE:  0.5179132608363675
IoU: 0.34944868417663255
Sensitivity:  0.9821818472805978
Specificity 0.9966983000195301
ACC:  0.9966718777743253
AUC:  0.9894400736500639
Prec:  0.3516781435430461
FScore:  0.5179132608363675 
#### Train Last
DICE: 0.8640
#### Test Last
DICE:  0.284989673611532
IoU: 0.16617373623147433
Sensitivity:  0.9775091346653554
Specificity 0.9957608781315929
ACC:  0.9957450450550426
AUC:  0.9866350063984741
Prec:  0.1668115204462133
FScore:  0.284989673611532
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dt = pd.read_json("dice_bce_history_masked_lung_500epc.json")

plt.plot(dt['dice_coef'])
plt.plot(dt['val_dice_coef'])
plt.plot(dt['loss'])
plt.title('Loss ')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train Dice', 'Val Dice', 'Loss'], loc='upper left')
# save plot to file
plt.savefig('dice_bce_plot_loss_masked_lung_500epc.png')
plt.show()