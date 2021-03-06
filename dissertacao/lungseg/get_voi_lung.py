# GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import traceback
import glob

from skimage.measure import label   

def getLargestCC(segmentation):
    labels = label(segmentation)
    unique, counts = np.unique(labels, return_counts=True)
    list_seg=list(zip(unique, counts))[1:] # the 0 label is by default background so take the rest
    
    largest_1=max(list_seg, key=lambda x:x[1])[0]
    del(list_seg[largest_1 - 1])
    largest_2=max(list_seg, key=lambda x:x[1])[0]
    
    label_max_1=(labels == largest_1).astype(int)
    label_max_2=(labels == largest_2).astype(int)
    return label_max_1, label_max_2

def get_voi_lung(exam_id, path_mask, output_path):
    try:
        print(exam_id + ':')

        image_mask = sitk.ReadImage(path_mask)
        image_array = sitk.GetArrayFromImage(image_mask).astype(np.int16)

        roi_1, roi_2 = getLargestCC(image_array)

        itkImage = sitk.GetImageFromArray((roi_1 + roi_2).astype(np.int16))
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
        output_path = dst_dir + '/' + exam_id + '_LungBBox' + ext

        # verifica se o arquivo ja existe
        if os.path.isfile(output_path):
            print('Arquivo ' + output_path + ' ja existe.')
            continue

        exam_ids.append(exam_id)
        input_src_paths.append(input_path)
        output_paths.append(output_path)
    
    for input_mask_path in input_mask_pathAll:
        input_mask_paths.append(input_mask_path)

    for i, exam_id in enumerate(tqdm(exam_ids,desc=desc)):
        get_voi_lung(exam_id, input_mask_paths[i], output_paths[i])

def main():
    ext = '.nii.gz'
    main_dir = '/home/anatielsantos/mestrado/datasets/dissertacao/dataset1/PulmoesZeroPedding' 
    main_mask_dir = '/home/anatielsantos/mestrado/datasets/dissertacao/dataset1/PulmoesZeroPedding/PulmoesMascaraUNet'
    
    mask_dir = '{}'.format(main_mask_dir)
    dst_dir = '{}/BBox'.format(main_dir)

    exec_get_voi_lung(mask_dir, dst_dir, ext, reverse = False, desc = 'Getting bounding box')

if __name__=="__main__":    
    main()