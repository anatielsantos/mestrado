# GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import SimpleITK as sitk
import glob, time
from tqdm import tqdm
import traceback
from skimage.morphology import ball

def lung_extract(exam_id, src_path, mask_path, output_path):
    try:
        print(exam_id + ':')
        
        image = sitk.ReadImage(src_path)
        npyImage = sitk.GetArrayFromImage(image)

        mask = sitk.ReadImage(mask_path)
        npyMask = sitk.GetArrayFromImage(mask)

        # postivate values if negative values
        imgMin = np.amin(npyImage)
        npyImage_aux = npyImage
        if (imgMin < 0):
            imgMin = imgMin * -1
            npyImage_aux = npyImage + imgMin

        newImage = npyImage_aux * npyMask

        itkImage = sitk.GetImageFromArray(newImage)
        sitk.WriteImage(itkImage, output_path)

        del image

    except Exception as e:
        print("type error: " + str(e))
        print(traceback.format_exc())
        return


def exec_lung_extract(src_dir, mask_dir, dst_dir, ext, reverse = False, desc = None):
    try:
        os.stat(dst_dir)
    except:
        os.mkdir(dst_dir)    

    input_src_pathAll = glob.glob(src_dir + '/*' + ext)
    input_src_pathAll.sort(reverse=reverse) 

    input_mask_pathAll = glob.glob(mask_dir + '/*' + ext)
    input_mask_pathAll.sort(reverse=reverse) 

    exam_ids = []
    input_src_paths = []
    input_mask_paths = []
    output_paths = []

    for input_path in input_src_pathAll:
        exam_id = os.path.basename(input_path.replace(ext, ''))
        output_path = dst_dir + '/' + exam_id + ext

        # verifica se o arquivo ja existe
        if os.path.isfile(output_path):
            print('Arquivo ' + output_path + ' ja existe e será removido')
            os.remove(output_path)
            # continue

        exam_ids.append(exam_id)
        input_src_paths.append(input_path)
        output_paths.append(output_path)
    
    for input_mask_path in input_mask_pathAll:
        input_mask_paths.append(input_mask_path)

    for i, exam_id in enumerate(tqdm(exam_ids,desc=desc)):
        lung_extract(exam_id, input_src_paths[i], input_mask_paths[i], output_paths[i])


def main():
    dataset = 'dataset1'
    ext = '.nii.gz'
    main_dir = f'/home/anatielsantos/mestrado/datasets/dissertacao/{dataset}/image'
    main_mask_dir = f'/home/anatielsantos/mestrado/datasets/dissertacao/{dataset}/lung_mask'
    
    src_dir = '{}'.format(main_dir)
    dst_dir = '{}/lung_extracted'.format(main_dir)

    mask_dir = '{}'.format(main_mask_dir)

    exec_lung_extract(src_dir, mask_dir, dst_dir, ext, reverse = False, desc = f'Extracting lungs')


if __name__=="__main__":    
    start = time.time()
    main()
    stop = time.time()
    print("Elapsed time: "+str(stop - start)+" seconds")