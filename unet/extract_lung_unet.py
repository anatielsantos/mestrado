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
import multiprocessing as mp
import traceback
import time
import multiprocessing.pool as mpp

from itertools import repeat
from skimage.feature import peak_local_max
from tqdm import tqdm
from train import unet

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

# import keras.models as models
# from skimage.transform import resize
# from skimage.io import imsave

np.random.seed(1337)
# tf.set_random_seed(1337)
tf.random.set_seed(1337)

# from keras.models import Model
# from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, AveragePooling2D, ZeroPadding3D
# from keras.optimizers import RMSprop, Adam, SGD
# from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
# from keras.layers import Dense,Flatten, Dropout,BatchNormalization, ZeroPadding2D,Lambda
from keras import backend as K
# from keras.regularizers import l2
# from keras.utils import plot_model
# from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator

# from skimage.transform import resize
# from skimage.io import imsave
# from sklearn.metrics import confusion_matrix

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

def load_patient(image):
    # imgs_train = np.load(image)
    # imgs_train = imgs_train['arr_0']

    # seed=1
    # datagen_images_train = ImageDataGenerator(
    #     featurewise_center=True,
    #     featurewise_std_normalization=True,
    #     )
    # datagen_images_train.fit(imgs_train)

    # del imgs_train

    itkImage = sitk.ReadImage(image)
    npyImage = sitk.GetArrayFromImage(itkImage)
    npyImage = np.expand_dims(npyImage, axis=-1)

    # for (X,_) in range(npyImage,[1]*npyImage.shape[0],batch_size=npyImage.shape[0], shuffle=False, seed=seed):

    #     return X
    #     break

    return npyImage

def predictPatient(model, image):

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    npyImage = load_patient(image)

    print('-'*30)
    print('Predicting test data...')
    print('-'*30)

    npyImagePredict = model.predict(npyImage, batch_size=1, verbose=1)
    
    npyImagePredict = preprocess_squeeze(npyImagePredict)
    npyImagePredict = np.around(npyImagePredict, decimals=0)
    npyImagePredict = (npyImagePredict>0.5)*1

    return npyImagePredict

def execExtractLungs(exam_id, input_path, output_path, model):
    try:
        print(exam_id + ':')

        binary_masks = predictPatient(model, input_path)

        # binary_masks = morphology.binary_fill_holes(
        #     morphology.binary_dilation(
        #         morphology.binary_fill_holes(binary_masks > 0),
        #         iterations=1)
        #     )

        # binary_masks.dtype='uint16'
        itkImage = sitk.GetImageFromArray(binary_masks)

        image = sitk.ReadImage(input_path)
        
        itkImage = sitk.Cast(itkImage,image.GetPixelIDValue())
        
        itkImage.CopyInformation(image)

        depth = image.GetDepth()
        step = depth//10
        min = depth//2 + step
        max = depth - step
        
        sitk.WriteImage(itkImage, output_path)

        del image

    except Exception as e:
        print("type error: " + str(e))
        print(traceback.format_exc())
        return

"""# Main"""

def execExtractLungsByUnet(src_dir, dst_dir, ext, search_pattern, model, reverse = False, desc = None, parallel = True):
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
            print('Arquivo ' + output_path + ' ja existe')
            # os.remove(output_path)
            continue

        exam_ids.append(exam_id)
        input_paths.append(input_path)
        output_paths.append(output_path)

    if(parallel):
        with mp.Pool(mp.cpu_count()) as pool:
            for _ in tqdm(pool.istarmap(execExtractLungs, zip(exam_ids, input_paths, output_paths, repeat(model))),
                          total=len(exam_ids)):
                pass
    else:
        for i, exam_id in enumerate(tqdm(exam_ids,desc=desc)):
            execExtractLungs(exam_id, input_paths[i], output_paths[i], model)

def main():

    ext = '.nii.gz'
    search_pattern = '*'
    dataset = 'dataset2'

    main_dir = f'/home/anatielsantos/mestrado/datasets/dissertacao/{dataset}/image/ZeroPedding'
    model_path = '/home/anatielsantos/mestrado/models/extractlung/2D-Unet_lungs.h5'

    src_dir = '{}'.format(main_dir)
    dst_dir = '{}/PredsUnet'.format(main_dir)

    nproc = mp.cpu_count()
    print('Num Processadores = ' + str(nproc))

    model = unet(pretrained_weights = None,input_size = (640,640,1))
    model.load_weights(model_path)

    execExtractLungsByUnet(src_dir, dst_dir, ext, search_pattern, model, reverse = False, desc = 'Predicting (Unet)', parallel=False)

if __name__ == '__main__':
    start = time.time()
    main()
    stop = time.time()
    print("Elapsed time: ", str(stop - start))