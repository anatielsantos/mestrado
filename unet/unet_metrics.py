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
from losses import calc_metric
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

project_name = 'Unet CovidSeg'
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

def execMetrics(exam_id, input_path, input_mask_path):
    try:
        print(exam_id + ':')

        binary_masks = load_patient(input_path)
        npyMedMask = load_patient(input_mask_path)

        print(binary_masks.shape)
        print(npyMedMask.shape)

        # calc metrics
        print('-'*30)
        print('Calculating metrics...')
        dice, jaccard, sensitivity, specificity, accuracy, auc, prec, fscore = calc_metric(binary_masks.astype(int), npyMedMask.astype(int))
        print("DICE:", dice)
        print("IoU:", jaccard)
        print("Sensitivity:", sensitivity)
        print("Specificity:", specificity)
        print("ACC:", accuracy)
        print("AUC:", auc)
        print("Prec:", prec)
        print("FScore:", fscore)

    except Exception as e:
        print("type error: " + str(e))
        print(traceback.format_exc())
        return

def execExecMetricsByUnet(src_dir, mask_dir, dst_dir, ext, search_pattern, reverse = False, desc = None, parallel = True):

    input_pathAll = glob.glob(src_dir + '/' + search_pattern + ext)
    input_pathAll.sort(reverse=reverse)

    input_mask_pathAll = glob.glob(mask_dir + '/' + search_pattern + ext)
    input_mask_pathAll.sort(reverse=reverse)

    exam_ids = []
    input_paths = []
    input_mask_paths = []
    output_paths = []

    for input_path in input_pathAll:
        exam_id = os.path.basename(input_path.replace(ext, ''))
        output_path = dst_dir + '/' + exam_id + '_PredLungseg' + ext

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
        # for i in tqdm(p.starmap(execPredict, zip(exam_ids, input_paths, output_paths, repeat(), repeat(normalize_path))),desc=desc):
        #     pass
        with mp.Pool(mp.cpu_count()) as pool:
            for _ in tqdm(pool.istarmap(execMetrics, zip(exam_ids, input_paths, output_paths, output_paths, repeat())),
                          total=len(exam_ids)):
                pass
    else:
        for i, exam_id in enumerate(tqdm(exam_ids,desc=desc)):
            execMetrics(exam_id, input_paths[i], input_mask_paths[i])

def main():

    ext = '.nii.gz'
    search_pattern = '*'
    dataset = 'dataset2'

    # local
    main_dir = f'/home/anatielsantos/mestrado/datasets/dissertacao/{dataset}/image/equalizeHist/ZeroPedding/UnetLungsegExp3PredsBest/VoiPulmoesMascara'
    main_mask_dir = f'/home/anatielsantos/mestrado/datasets/dissertacao/{dataset}/image/ZeroPedding/PulmoesMascara/PulmoesMascaraFillHoles'

    # remote
    # main_dir = f'/data/flavio/anatiel/datasets/dissertacao/{dataset}/image'
    # main_mask_dir = f'/data/flavio/anatiel/datasets/dissertacao/{dataset}/mask'

    src_dir = '{}'.format(main_dir)
    mask_dir = '{}'.format(main_mask_dir)
    dst_dir = '{}/UnetLungsegExp1PredsBest'.format(main_dir)

    nproc = mp.cpu_count()
    print('Num Processadores = ' + str(nproc))

    execExecMetricsByUnet(src_dir, mask_dir, dst_dir, ext, search_pattern, reverse = False, desc = 'Metrics (UNet)', parallel=False)

if __name__ == '__main__':
    start = time.time()
    main()
    stop = time.time()
    print("Elapsed time: ", str(stop - start))