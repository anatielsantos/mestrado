import os
import SimpleITK as sitk
import numpy as np
import glob, time
import multiprocessing as mp
import traceback
import time
import multiprocessing.pool as mpp
from itertools import repeat
from tqdm import tqdm
from train_exp_ds1 import unet
from losses import calc_metric

# GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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
img_rows = 544
img_cols = 544
img_depth = 1
smooth = 1.
K = '3'

 
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

    # return npyImagePredict.astype(np.float32)
    return npyImagePredict


def execPredict(exam_id, input_path, input_mask_path, output_path, model):
    try:
        print(exam_id + ':')

        binary_masks = predictPatient(model, input_path)
        npyMedMask = load_patient(input_mask_path)


        # calc metrics
        print('-'*30)
        print('Calculating metrics...')
        dice, jaccard, sensitivity, specificity, accuracy, auc, prec, fscore = calc_metric(binary_masks, npyMedMask)
        print("DICE:", dice)
        print("IoU:", jaccard)
        print("Sensitivity:", sensitivity)
        print("Specificity:", specificity)
        print("ACC:", accuracy)
        print("AUC:", auc)
        print("Prec:", prec)
        print("FScore:", fscore)

        binary_masks = binary_masks.astype(np.float32)
        npyMedMask = npyMedMask.astype(np.float32)

        itkImage = sitk.GetImageFromArray(binary_masks)

        image = sitk.ReadImage(input_path)
        #npyImage = sitk.GetArrayFromImage(image)

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
    input_paths = []
    input_mask_paths = []
    output_paths = []

    for input_path in input_pathAll:
        exam_id = os.path.basename(input_path.replace(ext, ''))
        output_path = dst_dir + '/' + exam_id + '_pred_exp3_exam_4' + K + ext

        # verifica se o arquivo ja existe
        if os.path.isfile(output_path):
            print('Arquivo ' + output_path + ' ja existe')
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
        print(str(len(input_paths)) + " - " + str(len(input_mask_paths)) + " - " + str(len(output_paths)))
        for i, exam_id in enumerate(tqdm(exam_ids,desc=desc)):
            execPredict(exam_id, input_paths[i], input_mask_paths[i], output_paths[i], model)


def main():

    ext = '.nii.gz'
    search_pattern = '*'
    dataset = 'dataset2'

    # remote   
    main_dir = f'/data/flavio/anatiel/datasets/dissertacao/final_tests/kfold/{dataset}/images/k{K}'
    main_mask_dir = f'/data/flavio/anatiel/datasets/dissertacao/final_tests/kfold/{dataset}/masks/k{K}'
    model_path = f'/data/flavio/anatiel/models/models_ds1/unet_ds1_150epc_best.h5'

    # main_dir = f'/home/anatielsantos/mestrado/datasets/dissertacao/bbox/{dataset}/images'
    # main_mask_dir = f'/home/anatielsantos/mestrado/datasets/dissertacao/bbox/{dataset}/masks'
    # model_path = '/home/anatielsantos/Downloads/models_dissertacao/models_ds1/unet_ds1_150epc_best.h5'

    src_dir = '{}'.format(main_dir)
    mask_dir = '{}'.format(main_mask_dir)
    dst_dir = '{}/unet_ds2_preds'.format(main_dir)

    nproc = mp.cpu_count()
    print('Num Processadores = ' + str(nproc))

    model = unet()
    model.load_weights(model_path)

    execExecPredictByUnet(src_dir, mask_dir, dst_dir, ext, search_pattern, model, reverse = False, desc = 'Predicting (UNet)', parallel=False)


if __name__ == '__main__':
    start = time.time()
    main()
    stop = time.time()
    print("Elapsed time: ", str(stop - start))
