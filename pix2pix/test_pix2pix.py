# GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf

import SimpleITK as sitk
import numpy as np
import glob, os, time
import multiprocessing as mp
from itertools import repeat
import traceback

from tqdm import tqdm

from model import Pix2Pix
from utils import *
from losses import *

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

np.random.seed(1337)
tf.random.set_seed(1337)

project_name = 'Pix2pix'
img_rows = 640
img_cols = 640
img_depth = 1
smooth = 1.

def preprocess_squeeze(imgs):
    imgs = np.squeeze(imgs, axis=3)
    print(' ---------------- preprocessed squeezed -----------------')
    return imgs

def load_images(path_src):
    src_npz = np.load(path_src, allow_pickle=True)
    src = src_npz['arr_0']
    
    return np.float32(src)

def load_patient(image):
    itkImage = sitk.ReadImage(image)
    npyImage = sitk.GetArrayFromImage(itkImage)
    # npyImage = np.expand_dims(npyImage, axis=-1)

    return npyImage

def predictPatient(model, image):

    print('-'*30)
    print('Loading test data...')
    npyImage = load_patient(image)

    print('-'*30)
    print("Saving npz file...")
    list_image = list()
    for i in range(npyImage.shape[0]):
        # images, masks = load_image(input_src_paths[i], input_mask_paths[i], remove_no_lesion=remove_no_lesion)
        list_image.append(npyImage[i])

    np.savez_compressed("/home/anatielsantos/mestrado/datasets/dissertacao/test/image/GanPredsLast/exam", list_image)

    print('-'*30)
    print("Loading npz file...")
    npzImage = load_images("/home/anatielsantos/mestrado/datasets/dissertacao/test/image/GanPredsLast/exam.npz")

    print('-'*30)
    print('Predicting test data...')
    print('-'*30)

    npzImagePredict=None
    for i in range(npzImage.shape[0]):
        pred = model.generator(npzImage[i:i+1], training=False).numpy()
        if npzImagePredict is None:
            npzImagePredict=pred
        else:
            npzImagePredict = np.concatenate([npzImagePredict,pred],axis=0)
    
    # npyImagePredict = preprocess_squeeze(npyImagePredict)
    # npyImagePredict = np.around(npyImagePredict, decimals=0)
    # npyImagePredict = (npyImagePredict>0.5)*1

    return np.asarray(npzImagePredict)

def execPredict(exam_id, input_path, output_path, model):
    try:
        print(exam_id + ':')

        binary_masks = predictPatient(model, input_path)

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
        # itkImage = sitk.Cast(itkImage,image.GetPixelIDValue())    
        
        itkImage.CopyInformation(image)

        # print(itkImage.GetSize())
        # print(itkImage.GetSpacing())
        # print(itkImage.GetPixelIDTypeAsString())    
        
        sitk.WriteImage(itkImage, output_path)

        del image

    except Exception as e:
        print("type error: " + str(e))
        print(traceback.format_exc())
        return

"""# Main"""

def execExecPredictByGan(src_dir, dst_dir, ext, search_pattern, model, reverse = False, desc = None, parallel = True):
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
        output_path = dst_dir + '/' + exam_id + '_Pred_teste' + ext

        # verifica se o arquivo ja existe
        if os.path.isfile(output_path):
            print('Arquivo ' + output_path + ' ja existe')
            # os.remove(output_path)
            continue

        exam_ids.append(exam_id)
        input_paths.append(input_path)
        output_paths.append(output_path)

    if(parallel):
        # p = mp.Pool(mp.cpu_count())
        # for i in tqdm(p.starmap(execPredict, zip(exam_ids, input_paths, output_paths, repeat(model), repeat(normalize_path))),desc=desc):
        #     pass
        with mp.Pool(mp.cpu_count()) as pool:
            for _ in tqdm(pool.istarmap(execPredict, zip(exam_ids, input_paths, output_paths, repeat(model))),
                          total=len(exam_ids)):
                pass
    else:
        for i, exam_id in enumerate(tqdm(exam_ids,desc=desc)):
            execPredict(exam_id, input_paths[i], output_paths[i], model)

def main():

    ext = '.nii.gz'
    search_pattern = '*'
    dataset = 'test'

    # local
    main_dir = f'/home/anatielsantos/mestrado/datasets/dissertacao/{dataset}/image'
    model_path = '/home/anatielsantos/mestrado/models/dissertacao/gan/gan_500epc_last.hdf5'

    # remote
    # main_dir = f'/data/flavio/anatiel/datasets/dissertacao/{dataset}/image/teste'
    # model_path = '/data/flavio/anatiel/models/dissertacao/gan_500epc_last.hdf5'

    src_dir = '{}'.format(main_dir)
    dst_dir = '{}/GanPredsLast'.format(main_dir)

    nproc = mp.cpu_count()
    print('Num Processadores = ' + str(nproc))

    model = Pix2Pix(img_rows, img_cols, img_depth, img_depth)
    model.compile(
        discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
        generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
        discriminator_loss = discriminator_loss,
        generator_loss = generator_loss
    )
    model.load_weights(model_path)

    execExecPredictByGan(src_dir, dst_dir, ext, search_pattern, model, reverse = False, desc = 'Predicting (Pix2pix)', parallel=False)

if __name__ == '__main__':
    start = time.time()
    main()
    stop = time.time()
    print("Elapsed time: ", (stop - start))