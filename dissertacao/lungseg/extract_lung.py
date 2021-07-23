# GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import SimpleITK as sitk
import glob, time
from tqdm import tqdm
import traceback

# extract pulmonary parenchyma
def extract_lung(exam_id, src_path, mask_path, output_path):
    try:
        print(exam_id + ':')
        
        image = sitk.ReadImage(src_path)
        npyImage = sitk.GetArrayFromImage(image)
        mask = sitk.ReadImage(mask_path)
        npyMask = sitk.GetArrayFromImage(mask)

        # new_image = npyImage
        # for s in range(len(new_image[:,0,0])):
        #     if s % 10 == 0:
        #         print(s, "/", len(new_image[:,0,0]))
        #     for l in range(len(new_image[0,:,0])):
        #         for c in range(len(new_image[0,0,:])):
        #             if (npyMask[s,l,c] < 1):
        #                 new_image[s,l,c] = 0
        
        itkImage = sitk.GetImageFromArray((npyImage * npyMask))
        sitk.WriteImage(itkImage, output_path)

        del image

    except Exception as e:
        print("type error: " + str(e))
        print(traceback.format_exc())
        return
    
def exec_extract_lung(src_dir, mask_dir, dst_dir, ext, reverse = False, desc = None):
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
        output_path = dst_dir + '/' + exam_id + '_LungExtract' + ext

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
        extract_lung(exam_id, input_src_paths[i], input_mask_paths[i], output_paths[i])
            
def main():
    dataset = 'dataset1'
    ext = '.nii.gz'
    main_dir = f'/home/anatielsantos/mestrado/datasets/dissertacao/{dataset}/PulmoesZeroPedding' 
    main_mask_dir = f'/home/anatielsantos/mestrado/datasets/dissertacao/{dataset}/PulmoesZeroPedding/PulmoesMascaraFillHoles'
    
    src_dir = '{}'.format(main_dir)
    dst_dir = '{}/Pulmoes'.format(main_dir)

    mask_dir = '{}'.format(main_mask_dir)

    exec_extract_lung(src_dir, mask_dir, dst_dir, ext, reverse = False, desc = f'Extracting lung from {dataset}')

if __name__=="__main__":    
    # arquivo = open("anatiel/dissertacao/lungseg/time_execution_zero_padding.txt", "a")
    # start = time.time()
    main()
    # stop = time.time()
    # exec_time = stop - start
    # arquivo.write(str(exec_time))
