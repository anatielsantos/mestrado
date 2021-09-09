# GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import SimpleITK as sitk
import traceback
import glob
from tqdm import tqdm
from scipy.ndimage import morphology

from skimage.measure import label

def fill_holes(binary_masks):
    # with structure element
    binary_masks = morphology.binary_fill_holes(
    morphology.binary_dilation(
        morphology.binary_fill_holes(binary_masks > 0),
        iterations=1), structure=np.ones((3,1,1))
    ).astype(np.int)

    # without structure element
    # binary_masks = morphology.binary_fill_holes(
    # morphology.binary_dilation(
    #     morphology.binary_fill_holes(binary_masks > 0),
    #     iterations=1)
    # )

    return binary_masks

def getLargestCC(segmentation):
    labels = label(segmentation)
    unique, counts = np.unique(labels, return_counts=True)
    list_seg=list(zip(unique, counts))[1:] # the 0 label is by default background so take the rest
    
    largest_1=max(list_seg, key=lambda x:x[1])[0]
    del(list_seg[largest_1 - 1])
    largest_2=max(list_seg, key=lambda x:x[1])[0]
    
    label_max_1=(labels == largest_1).astype(int)
    label_max_2=(labels == largest_2).astype(int)

    return (label_max_1 + label_max_2)

def get_voi_lung(exam_id, path_mask, output_path):
    try:
        print(exam_id + ':')

        image_mask = sitk.ReadImage(path_mask)
        image_array = sitk.GetArrayFromImage(image_mask).astype(np.int16)

        voi = getLargestCC(image_array)

        voi = fill_holes(voi)

        itkImage = sitk.GetImageFromArray(voi.astype(np.int16))
        sitk.WriteImage(itkImage, output_path)

        del image_mask

    except Exception as e:
        print("type error: " + str(e))
        print(traceback.format_exc())
        return

def exec_get_voi_lung(mask_dir, dst_dir, ext, reverse = False, desc = None):
    try:
        os.stat(dst_dir)
    except:
        os.mkdir(dst_dir)    

    input_mask_pathAll = glob.glob(mask_dir + '/*' + ext)
    input_mask_pathAll.sort(reverse=reverse) 

    exam_ids = []
    input_src_paths = []
    input_mask_paths = []
    output_paths = []

    for input_path in input_mask_pathAll:
        exam_id = os.path.basename(input_path.replace(ext, ''))
        output_path = dst_dir + '/' + exam_id + '_FillHoles' + ext

        # verifica se o arquivo ja existe
        if os.path.isfile(output_path):
            print('Arquivo ' + output_path + ' ja existe')
            # os.remove(output_path)
            continue

        exam_ids.append(exam_id)
        input_src_paths.append(input_path)
        output_paths.append(output_path)
    
    for input_mask_path in input_mask_pathAll:
        input_mask_paths.append(input_mask_path)

    for i, exam_id in enumerate(tqdm(exam_ids,desc=desc)):
        get_voi_lung(exam_id, input_mask_paths[i], output_paths[i])

def main():
    dataset = 'dataset2'
    ext = '.nii.gz' 
    main_mask_dir = f'/home/anatielsantos/mestrado/datasets/dissertacao/{dataset}/image/ZeroPedding/UnetLungsegExp1PredsBest'
    
    mask_dir = '{}'.format(main_mask_dir)
    dst_dir = '{}/VoiPulmoesMascara'.format(main_mask_dir)

    exec_get_voi_lung(mask_dir, dst_dir, ext, reverse = False, desc = f'Getting VOI from {dataset}')

if __name__=="__main__":    
    main()