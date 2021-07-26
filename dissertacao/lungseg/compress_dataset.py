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
def compress_dataset(exam_id, path):
    try:
        print(exam_id + ':')
        
        image = sitk.ReadImage(path)
        npyImage = sitk.GetArrayFromImage(image)

        del image
        return npyImage

    except Exception as e:
        print("type error: " + str(e))
        print(traceback.format_exc())
        return
    
def exec_compress_dataset(src_dir, dst_dir, ext, reverse = False, desc = None):
    try:
        os.stat(dst_dir)
    except:
        os.mkdir(dst_dir)    

    input_src_pathAll = glob.glob(src_dir + '/*' + ext)
    input_src_pathAll.sort(reverse=reverse)

    exam_ids = []
    input_src_paths = []

    output_path = dst_dir
    joint = 'test' # [train, val, test]
    
    for input_path in input_src_pathAll:
        exam_id = os.path.basename(input_path.replace(ext, ''))

        exam_ids.append(exam_id)
        input_src_paths.append(input_path)

    images = []
    for i, exam_id in enumerate(tqdm(exam_ids,desc=desc)):
        images.append(compress_dataset(exam_id, input_src_paths[i]))
    
    np.savez_compressed(f"{output_path}/{joint}",images)
            
def main():
    ext = '.nii.gz'
    im = 'masks'
    main_dir_train = f'/home/anatielsantos/mestrado/datasets/dissertacao/{im}/train'
    main_dir_val = f'/home/anatielsantos/mestrado/datasets/dissertacao/{im}/val'
    main_dir_test = f'/home/anatielsantos/mestrado/datasets/dissertacao/{im}/test'
    
    src = main_dir_test
    dst = main_dir_test
    src_dir = '{}'.format(src)
    dst_dir = '{}'.format(dst)

    exec_compress_dataset(src_dir, dst_dir, ext, reverse = False, desc = f'Compressing datasets')

if __name__=="__main__":    
    start = time.time()
    main()
    stop = time.time()
    print("Elapsed time: "+str(stop - start)+" seconds")
