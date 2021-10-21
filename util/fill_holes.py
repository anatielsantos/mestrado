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
from skimage.morphology import ball, disk

def fill_holes(binary_masks):
    elementStructure = ball(3, dtype=np.uint8)
    # elementStructure = np.transpose(np.expand_dims(elementStructure, axis=-1), (2, 0, 1))
    # binary_masks = morphology.binary_erosion(
    #         morphology.binary_dilation(
    #             morphology.binary_fill_holes(
    #                 binary_masks > 0
    #             ), iterations=1, structure=elementStructure
    #         ), structure=elementStructure
    #     )

    # fh = morphology.binary_fill_holes(binary_masks)
    binary_masks = morphology.binary_closing(
                binary_masks, iterations=2, structure=elementStructure
            )
    

    return binary_masks

def get_voi_lung(exam_id, path_mask, output_path, bbox=False):
    try:
        print(exam_id + ':')

        image_mask = sitk.ReadImage(path_mask)
        image_array = sitk.GetArrayFromImage(image_mask).astype(np.int16)

        image_array_fh = fill_holes(image_array)

        image_array_fh = abs(image_array - image_array_fh)
        image_array_fh = image_array + image_array_fh
        image_array_fh[image_array_fh[:,:] > 1] = 1

        itkImage = sitk.GetImageFromArray(image_array_fh.astype(np.int16))
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
        get_voi_lung(exam_id, input_mask_paths[i], output_paths[i], bbox=False)

def main():
    dataset = 'dataset2'
    ext = '.nii.gz' 
    main_mask_dir = f'/home/anatielsantos/mestrado/datasets/dissertacao/{dataset}/image/ZeroPedding/imagePositive/VoiLung/bk_VoiLungFillHoles'
    mask_dir = '{}'.format(main_mask_dir)
    dst_dir = '{}/VoiLungFillHoles'.format(main_mask_dir)

    exec_get_voi_lung(mask_dir, dst_dir, ext, reverse = False, desc = f'Getting VOI from {dataset}')

if __name__=="__main__":
    main()