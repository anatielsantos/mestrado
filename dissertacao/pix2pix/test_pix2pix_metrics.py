# GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf

import SimpleITK as sitk
import numpy as np
import glob, time
import multiprocessing as mp
import traceback
import time
import multiprocessing.pool as mpp
from itertools import repeat

from tqdm import tqdm
from model import Pix2Pix
from losses import *
from skimage.exposure import rescale_intensity

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

np.random.seed(1337)

# train settings
project_name = 'Pix2pix LungSeg'
BUFFER_SIZE = 400
BATCH_SIZE = 1
EPOCHS = 150
IMG_WIDTH = 544
IMG_HEIGHT = 544
INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 1


def preprocess_squeeze(imgs):
    imgs = np.squeeze(imgs, axis=3)
    print('preprocessed squeezed')
    print('-'*30)
    return imgs


def load_patient(image):
    itkImage = sitk.ReadImage(image)
    npyImage = sitk.GetArrayFromImage(itkImage)
    npyImage = np.expand_dims(npyImage, axis=-1)

    return npyImage


def predictPatient(model, image):

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    npyImage = load_patient(image)

    print("Image shape:", npyImage.shape)

    print('-'*30)
    print('Predicting test data...')
    print('-'*30)

    # npyImagePredict = model.predict(npyImage, batch_size=1)
    npyImagePredict=None
    for i in range(npyImage.shape[0]):
        pred = model.generator(npyImage[i:i+1], training=False).numpy()
        if npyImagePredict is None:
            npyImagePredict=pred
        else:
            npyImagePredict = np.concatenate([npyImagePredict,pred],axis=0)

    npyImagePredict = preprocess_squeeze(npyImagePredict)
    # npyImagePredict = np.around(npyImagePredict, decimals=0)
    # npyImagePredict = (npyImagePredict>0.5)*1

    return npyImagePredict.astype(np.float32)


def execPredict(exam_id, input_path, input_mask_path, output_path, model):
    try:
        print(exam_id + ':')

        binary_masks = predictPatient(model, input_path)
        npyMedMask = load_patient(input_mask_path)

        print("Pred shape:", binary_masks.shape)
        print("Mask shape:", npyMedMask.shape)

        # calc metrics
        print('-'*30)
        print('Calculating metrics...')
        dice, jaccard, sensitivity, specificity, accuracy, auc, prec, fscore = calc_metric(binary_masks.astype(int), npyMedMask.astype(int))
        print("DICE:\t", dice)
        print("IoU:\t", jaccard)
        print("Sensitivity:\t", sensitivity)
        print("Specificity:\t", specificity)
        print("ACC:\t", accuracy)
        print("AUC:\t", auc)
        print("Prec:\t", prec)
        print("FScore:\t", fscore)

        # binary_masks.dtype='float32'
        itkImage = sitk.GetImageFromArray(binary_masks)
        image = sitk.ReadImage(input_path)
        itkImage.CopyInformation(image)
        sitk.WriteImage(itkImage, output_path)

        del image

    except Exception as e:
        print("type error: " + str(e))
        print(traceback.format_exc())
        return


def execExecPredictByUnet(src_dir, mask_dir, dst_dir, ext, search_pattern, model, reverse = False, desc = None, parallel = True):
    try:
        os.stat(dst_dir)
    except:
        os.mkdir(dst_dir)

    input_pathAll = glob.glob(src_dir + '/' + search_pattern + ext)
    input_pathAll.sort(reverse=reverse)

    input_mask_pathAll = glob.glob(mask_dir + '/' + search_pattern + ext)
    input_mask_pathAll.sort(reverse=reverse)

    exam_ids = []
    mask_ids = []
    input_paths = []
    input_mask_paths = []
    output_paths = []

    for input_path in input_pathAll:
        exam_id = os.path.basename(input_path.replace(ext, ''))
        output_path = dst_dir + '/' + exam_id + '_pred_k2' + ext

        # verifica se o arquivo ja existe
        if os.path.isfile(output_path):
            print('Arquivo ' + output_path + ' ja existe')
            # os.remove(output_path)
            continue

        exam_ids.append(exam_id)
        input_paths.append(input_path)
        output_paths.append(output_path)

    for input_mask_path in input_mask_pathAll:
        input_mask_paths.append(input_mask_path)

    if(parallel):
        # p = mp.Pool(mp.cpu_count())
        # for i in tqdm(p.starmap(execPredict, zip(exam_ids, input_paths, output_paths, repeat(model), repeat(normalize_path))),desc=desc):
        #     pass
        with mp.Pool(mp.cpu_count()) as pool:
            for _ in tqdm(pool.istarmap(execPredict, zip(exam_ids, input_paths, output_paths, output_paths, repeat(model))),
                          total=len(exam_ids)):
                pass
    else:
        for i, exam_id in enumerate(tqdm(exam_ids,desc=desc)):
            execPredict(exam_id, input_paths[i], input_mask_paths[i], output_paths[i], model)


def main():

    ext = '.nii.gz'
    search_pattern = '*'
    dataset = 'dataset1'
    KF = '0'

    # remote
    main_dir = f'/data/flavio/anatiel/datasets/dissertacao/final_tests/kfold/{dataset}/images/k{KF}/'
    main_mask_dir = f'/data/flavio/anatiel/datasets/dissertacao/final_tests/kfold/{dataset}/masks/k{KF}/'
    model_path = f'/data/flavio/anatiel/models/models_kfold/gan_ds1_150epc_best_k{KF}.hdf5'

    # main_dir = f'/home/anatielsantos/mestrado/datasets/dissertacao/bbox/dataset1/images/k0'
    # main_mask_dir = f'/home/anatielsantos/mestrado/datasets/dissertacao/bbox/dataset1/masks/k0'
    # model_path = '/home/anatielsantos/Downloads/models_dissertacao/gan_ds1_150epc_best_k0.hdf5'

    src_dir = '{}'.format(main_dir)
    mask_dir = '{}'.format(main_mask_dir)
    dst_dir = '{}gan_preds'.format(main_dir)

    nproc = mp.cpu_count()
    print('Num Processadores = ' + str(nproc))

    model = Pix2Pix(IMG_HEIGHT,IMG_WIDTH,INPUT_CHANNELS,OUTPUT_CHANNELS)
    model.compile(
        discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
        generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
        discriminator_loss = discriminator_loss,
        generator_loss = generator_loss
    )
    model.load_weights(model_path)

    execExecPredictByUnet(src_dir, mask_dir, dst_dir, ext, search_pattern, model, reverse = False, desc = 'Predicting (GAN)', parallel=False)


if __name__ == '__main__':
    start = time.time()
    main()
    stop = time.time()
    print("Elapsed time: ", str(stop - start))
