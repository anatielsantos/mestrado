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
from lesionseg.train import unet
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

def get_bbox(mask, pred_mask):
    new_image = np.zeros(pred_mask.shape)
    for s in range(len(new_image[:,0,0])):
        menor_c =  640
        maior_c =  0
        menor_l =  640
        maior_l =  0
        print(s, "/", len(mask[:,0,0]))
        for l in range(len(mask[0,:,0])):
            for c in range(len(mask[0,0,:])):
                if ((mask[s,l,c] == 1) & (c < menor_c)):
                    menor_c = c
                if ((mask[s,l,c] == 1) & (l < menor_l)):
                    menor_l = l
                if ((mask[s,l,c] == 1) & (c > maior_c)):
                    maior_c = c
                if ((mask[s,l,c] == 1) & (l > maior_l)):
                    maior_l = l
                
        new_image[s, menor_l+5:maior_l-4,menor_c+5:maior_c-4] = pred_mask[s, menor_l+5:maior_l-4,menor_c+5:maior_c-4]
    
    return np.expand_dims(new_image, axis=-1)
    
def predictPatient(model, image):

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    npyImage = load_patient(image)

    # Normalization of the train set (Exp 1)
    npyImage = npyImage.astype('float32')
    mean = np.mean(npyImage)  # mean for data centering
    std = np.std(npyImage)  # std for data normalization
    npyImage -= mean
    npyImage /= std

    # npyImage = rescale_intensity(npyImage, in_range=(0, 1))
    # npyImage = npyImage.astype('float32')

    print('-'*30)
    print('Predicting test data...')
    print('-'*30)

    npyImagePredict = model.predict(npyImage, batch_size=1, verbose=1)
    
    npyImagePredict = preprocess_squeeze(npyImagePredict)
    npyImagePredict = np.around(npyImagePredict, decimals=0)
    npyImagePredict = (npyImagePredict>0.5)*1

    return npyImagePredict.astype(np.float32)

def execPredict(exam_id, input_path, input_mask_path, output_path, model):
    try:
        print(exam_id + ':')

        binary_masks = predictPatient(model, input_path)
        npyMedMask = load_patient(input_mask_path)

        # binary_masks = get_bbox(npyMedMask, binary_masks)

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


        # binary_masks.dtype='float32'
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
        output_path = dst_dir + '/' + exam_id + '_PredLesionSeg_2_2_last' + ext

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
        print(str(len(input_paths)) + " - " + str(len(input_mask_paths)) + " - " + str(len(output_paths)))
        for i, exam_id in enumerate(tqdm(exam_ids,desc=desc)):
            execPredict(exam_id, input_paths[i], input_mask_paths[i], output_paths[i], model)

def main():

    ext = '.nii.gz'
    search_pattern = '*'
    dataset = 'dataset2'

    # local
    # main_dir = f'/home/anatielsantos/mestrado/datasets/dissertacao/final_tests_dis/teste2_dataset2/Test'
    # main_mask_dir = f'/home/anatielsantos/mestrado/datasets/dissertacao/final_tests_dis/teste2_dataset2/Test_mask'
    # model_path = '/home/anatielsantos/mestrado/datasets/dissertacao/final_tests_dis/original_unet_exp2_2_last.h5'

    # remote
    main_dir = f'/data/flavio/anatiel/datasets/dissertacao/final_tests/tests'
    main_mask_dir = f'/data/flavio/anatiel/datasets/dissertacao/final_tests/tests/Test_mask'
    model_path = '/data/flavio/anatiel/models/dissertacao/final_tests/original_unet_exp4_1_best.h5'

    src_dir = '{}'.format(main_dir)
    mask_dir = '{}'.format(main_mask_dir)
    dst_dir = '{}/PredsOriginal'.format(main_dir)

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