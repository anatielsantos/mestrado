# GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow as tf

import sys
import SimpleITK as sitk
import numpy as np
import glob, os, time
# import statistics as stat
# import threading
# from scipy.ndimage import morphology
# from scipy import signal
# from matplotlib import pyplot
import multiprocessing as mp
from itertools import repeat
import traceback
import time

# from scipy import ndimage as ndi
# from skimage.segmentation import watershed
from skimage.feature import peak_local_max

# if IN_NOTEBOOK:
#     from tqdm.notebook import tqdm
# else:
from tqdm import tqdm

"""# Definição de funções"""

import multiprocessing.pool as mpp

def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    """
    if self._state != mpp.RUN:
        raise ValueError("Pool not running")

    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self._cache)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)

mpp.Pool.istarmap = istarmap


import keras.models as models
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import SimpleITK as sitk
import glob

np.random.seed(1337)
# tf.set_random_seed(1337)
tf.random.set_seed(1337)

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, AveragePooling2D, ZeroPadding3D
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.layers import Dense,Flatten, Dropout,BatchNormalization, ZeroPadding2D,Lambda
from keras import backend as K
from keras.regularizers import l2
# from keras.utils import plot_model
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator

from skimage.transform import resize
from skimage.io import imsave
from sklearn.metrics import confusion_matrix

# from data import load_train_data, load_test_data, load_val_data

K.set_image_data_format('channels_last')

project_name = '2D-Unet'
img_rows = 640
img_cols = 640
img_depth = 1
smooth = 1.

def preprocess_squeeze(imgs):
    imgs = np.squeeze(imgs, axis=3)
    print(' ---------------- preprocessed squeezed -----------------')
    return imgs

def dice_coef(y_true, y_pred):
    im_sum = K.sum(y_pred) + K.sum(y_true)
    intersection = y_true * y_pred
    return 2.*K.sum(intersection)/im_sum

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def get_unet():
    inputs = Input((img_rows, img_cols, 1))

    conv1 = Conv2D(32, (3, 3), padding='same')(inputs)
    conv1 = LeakyReLU()(conv1)
    conv1 = BatchNormalization()(conv1)
    #conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), padding='same')(conv1)
    conv1 = LeakyReLU()(conv1)
    conv1 = BatchNormalization()(conv1)
    conc1 = concatenate([inputs, conv1], axis=3)
    pool1 = AveragePooling2D(pool_size=(2, 2))(conc1)


    conv2 = Conv2D(64, (3, 3), padding='same')(pool1)
    conv2 = LeakyReLU()(conv2)
    conv2 = BatchNormalization()(conv2)
    #conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), padding='same')(conv2)
    conv2 = LeakyReLU()(conv2)
    conv2 = BatchNormalization()(conv2)
    conc2 = concatenate([pool1, conv2], axis=3)
    pool2 = AveragePooling2D(pool_size=(2, 2))(conc2)


    conv3 = Conv2D(128, (3, 3), padding='same')(pool2)
    conv3 = LeakyReLU()(conv3)
    conv3 = BatchNormalization()(conv3)
    #conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), padding='same')(conv3)
    conv3 = LeakyReLU()(conv3)
    conv3 = BatchNormalization()(conv3)
    conc3 = concatenate([pool2, conv3], axis=3)
    pool3 = AveragePooling2D(pool_size=(2, 2))(conc3)


    conv4 = Conv2D(256, (3, 3), padding='same')(pool3)
    conv4 = LeakyReLU()(conv4)
    conv4 = BatchNormalization()(conv4)
    #conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(256, (3, 3), padding='same')(conv4)
    conv4 = LeakyReLU()(conv4)
    conv4 = BatchNormalization()(conv4)
    conc4 = concatenate([pool3, conv4], axis=3)
    pool4 = AveragePooling2D(pool_size=(2, 2))(conc4)


    conv5 = Conv2D(512, (3, 3), padding='same')(pool4)
    conv5 = LeakyReLU()(conv5)
    conv5 = BatchNormalization()(conv5)
    #conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(512, (3, 3), padding='same')(conv5)
    conv5 = LeakyReLU()(conv5)
    conv5 = BatchNormalization()(conv5)
    conc5 = concatenate([pool4, conv5], axis=3)


    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conc5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), padding='same')(up6)
    conv6 = LeakyReLU()(conv6)
    conv6 = BatchNormalization()(conv6)
    #conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(256, (3, 3), padding='same')(conv6)
    conv6 = LeakyReLU()(conv6)
    conv6 = BatchNormalization()(conv6)
    conc6 = concatenate([up6, conv6], axis=3)


    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conc6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), padding='same')(up7)
    conv7 = LeakyReLU()(conv7)
    conv7 = BatchNormalization()(conv7)
    #conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(128, (3, 3), padding='same')(conv7)
    conv7 = LeakyReLU()(conv7)
    conv7 = BatchNormalization()(conv7)
    conc7 = concatenate([up7, conv7], axis=3)


    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conc7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), padding='same')(up8)
    conv8 = LeakyReLU()(conv8)
    conv8 = BatchNormalization()(conv8)
    #conv8 = Dropout(0.2)(conv8)
    conv8 = Conv2D(64, (3, 3), padding='same')(conv8)
    conv8 = LeakyReLU()(conv8)
    conv8 = BatchNormalization()(conv8)
    conc8 = concatenate([up8, conv8], axis=3)


    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conc8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), padding='same')(up9)
    conv9 = LeakyReLU()(conv9)
    #conv9 = Dropout(0.2)(conv9)
    conv9 = Conv2D(32, (3, 3), padding='same')(conv9)
    conv9 = LeakyReLU()(conv9)
    conc9 = concatenate([up9, conv9], axis=3)


    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conc9)

    model = Model(inputs=[inputs], outputs=[conv10])

    #model.summary()
    #plot_model(model, to_file='model.png')

    model.compile(optimizer=Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199), loss=dice_coef_loss, metrics=['accuracy', dice_coef])

    return model


def load_patient(image, normalize_path):
    imgs_train = np.load(normalize_path)
    imgs_train = imgs_train['arr_0']

    seed=1    
    datagen_images_train = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        )
    datagen_images_train.fit(imgs_train)

    
    del imgs_train

    itkImage = sitk.ReadImage(image)
    npyImage = sitk.GetArrayFromImage(itkImage)
    npyImage = np.expand_dims(npyImage, axis=-1)

    for (X,_) in datagen_images_train.flow(npyImage,[1]*npyImage.shape[0],batch_size=npyImage.shape[0], shuffle=False, seed=seed):

        return X
        break


def predictPatient(model, image, normalize_path):

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    npyImage = load_patient(image, normalize_path)

    print('-'*30)
    print('Predicting test data...')
    print('-'*30)

    npyImagePredict = model.predict(npyImage, batch_size=1, verbose=1)
    
    npyImagePredict = preprocess_squeeze(npyImagePredict)
    npyImagePredict = np.around(npyImagePredict, decimals=0)
    npyImagePredict = (npyImagePredict>0.5)*1

    return npyImagePredict

def execExtractLungs(exam_id, input_path, output_path, model, normalize_path):
    try:
        print(exam_id + ':')

        binary_masks = predictPatient(model, input_path, normalize_path)

        # binary_masks = morphology.binary_fill_holes(
        #     morphology.binary_dilation(
        #         morphology.binary_fill_holes(binary_masks > 0),
        #         iterations=1)
        #     )

        # binary_masks.dtype='uint16'
        itkImage = sitk.GetImageFromArray(binary_masks)

        image = sitk.ReadImage(input_path)
        #npyImage = sitk.GetArrayFromImage(image)

        # print(image.GetSize())
        # print(image.GetSpacing())
        # print(image.GetPixelIDTypeAsString())
        # print(itkImage.GetSize())
        # print(itkImage.GetSpacing())
        # print(itkImage.GetPixelIDTypeAsString())

        # corrige o tipo da imagem pois no Colab está saindo 64 bits
        itkImage = sitk.Cast(itkImage,image.GetPixelIDValue())    
        
        itkImage.CopyInformation(image)

        # print(itkImage.GetSize())
        # print(itkImage.GetSpacing())
        # print(itkImage.GetPixelIDTypeAsString())    

        depth = image.GetDepth()
        step = depth//10
        min = depth//2 + step
        max = depth - step  

        # if(IN_NOTEBOOK):        
        #     myshow3d(img = sitk.LabelOverlay(sitk.RescaleIntensity(image), sitk.LabelContour(
        #         itkImage)), title='Mascara Pulmões(Unet)({})'.format(exam_id), zslices=range(min, max, step), dpi=80)  

        # save2d(output_path + '_2d.png', img = sitk.LabelOverlay(sitk.RescaleIntensity(image), sitk.LabelContour(
        #         itkImage)), title='Mascara Pulmões(Unet)({})'.format(exam_id), zslice=362)

        # save3d(output_path + '.png', img = sitk.LabelOverlay(sitk.RescaleIntensity(image), sitk.LabelContour(
        #     itkImage)), title='Mascara Pulmões(Unet)({})'.format(exam_id), zslices=range(min, max, step), dpi=80)    
        
        sitk.WriteImage(itkImage, output_path)

        del image

    except Exception as e:
        print("type error: " + str(e))
        print(traceback.format_exc())
        return

"""# Main"""

def execExtractLungsByUnet(src_dir, dst_dir, ext, search_pattern, model, normalize_path, reverse = False, desc = None, parallel = True):
    try:
        os.stat(dst_dir)
    except:
        os.mkdir(dst_dir)

    input_pathAll = glob.glob(src_dir + '/' + search_pattern + ext)
    input_pathAll.sort(reverse=reverse)

    exam_ids = []
    input_paths = []
    output_paths = []

    for input_path in input_pathAll:
        exam_id = os.path.basename(input_path.replace(ext, ''))
        output_path = dst_dir + '/' + exam_id + '_lungMask' + ext

        # verifica se o arquivo ja existe
        if os.path.isfile(output_path):
            print('Arquivo ' + output_path + ' ja existe e será removido')
            os.remove(output_path)
            # continue

        exam_ids.append(exam_id)
        input_paths.append(input_path)
        output_paths.append(output_path)

    if(parallel):
        # p = mp.Pool(mp.cpu_count())
        # for i in tqdm(p.starmap(execExtractLungs, zip(exam_ids, input_paths, output_paths, repeat(model), repeat(normalize_path))),desc=desc):
        #     pass
        with mp.Pool(mp.cpu_count()) as pool:
            for _ in tqdm(pool.istarmap(execExtractLungs, zip(exam_ids, input_paths, output_paths, repeat(model), repeat(normalize_path))),
                          total=len(exam_ids)):
                pass
    else:
        for i, exam_id in enumerate(tqdm(exam_ids,desc=desc)):
            execExtractLungs(exam_id, input_paths[i], output_paths[i], model, normalize_path)

def main():

    ext = '.nii.gz'
    search_pattern = '*'
    dataset = 'dataset2'

    main_dir = f'/home/anatielsantos/mestrado/datasets/dissertacao/{dataset}/PulmoesZeroPedding/'
    model_path = '/home/anatielsantos/mestrado/models/extractlung/2D-Unet_lungs.h5'
    normalize_path = '/home/anatielsantos/mestrado/models/extractlung/images_test_lungs.npz'

    src_dir = '{}'.format(main_dir)
    dst_dir = '{}/PulmoesMascara16bits'.format(main_dir)

    nproc = mp.cpu_count()
    print('Num Processadores = ' + str(nproc))

    model = get_unet()
    model.load_weights(model_path)

    execExtractLungsByUnet(src_dir, dst_dir, ext, search_pattern, model, normalize_path, reverse = False, desc = 'Extraindo pulmões (unet)', parallel=False)

if __name__ == '__main__':
    # arquivo = open("anatiel/dissertacao/lungseg/time_execution_lungseg.txt", "a")
    # start = time.time()
    main()
    # stop = time.time()
    # exec_time = stop - start
    # arquivo.write(str(exec_time))

    