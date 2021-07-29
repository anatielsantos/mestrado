# GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import SimpleITK as sitk
import numpy as np
import glob, time
import multiprocessing as mp
import traceback
import time
import multiprocessing.pool as mpp
from itertools import repeat

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

np.random.seed(1337)

project_name = 'Unet LungSeg'
img_rows = 640
img_cols = 640
img_depth = 1
smooth = 1.

def preprocess_squeeze(imgs):
    imgs = np.squeeze(imgs, axis=3)
    print(' ---------------- preprocessed squeezed -----------------')
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

    print('-'*30)
    print('Predicting test data...')
    print('-'*30)

    npyImagePredict = model.predict(npyImage, batch_size=1, verbose=1)
    
    npyImagePredict = preprocess_squeeze(npyImagePredict)
    npyImagePredict = np.around(npyImagePredict, decimals=0)
    npyImagePredict = (npyImagePredict>0.5)*1

    return npyImagePredict

def execPredictPatient(exam_id, input_path, output_path, model):
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
            for _ in tqdm(pool.istarmap(execPredictPatient, zip(exam_ids, input_paths, output_paths, repeat(model))),
                          total=len(exam_ids)):
                pass
    else:
        for i, exam_id in enumerate(tqdm(exam_ids,desc=desc)):
            execPredictPatient(exam_id, input_paths[i], output_paths[i], model)

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