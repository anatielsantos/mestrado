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
def load_image(exam_id, path):
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
    
def compress_dataset(src_dir, dst_dir, ext, reverse = False, desc = None):
    try:
        os.stat(dst_dir)
    except:
        os.mkdir(dst_dir)    

    input_src_pathAll = glob.glob(src_dir + '/*' + ext)
    input_src_pathAll.sort(reverse=reverse)

    exam_ids = []
    input_src_paths = []

    output_path = dst_dir
    joint = 'train_mask' # [train, val, test]
    
    for input_path in input_src_pathAll:
        exam_id = os.path.basename(input_path.replace(ext, ''))

        exam_ids.append(exam_id)
        input_src_paths.append(input_path)

    images = []
    for i, exam_id in enumerate(tqdm(exam_ids,desc=desc)):
        print(i)
        if i < 9:
            images.append(load_image(exam_id, input_src_paths[i]))

    np.savez_compressed(f"{output_path}/{joint}",np.asarray(images))
            
def main():
    ext = '.nii.gz'
    dataset = 'dataset1'
    im = 'mask'
    main_dir = f'/home/anatielsantos/mestrado/datasets/dissertacao/{dataset}/{im}'
    
    src = main_dir
    dst = main_dir
    src_dir = '{}'.format(src)
    dst_dir = '{}'.format(dst)

    compress_dataset(src_dir, dst_dir, ext, reverse = False, desc = f'Compressing datasets')

if __name__=="__main__":    
    start = time.time()
    main()
    stop = time.time()
    print("Elapsed time: "+str(stop - start)+" seconds")
