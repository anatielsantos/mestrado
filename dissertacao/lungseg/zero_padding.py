# GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import numpy as np
import SimpleITK as sitk
import glob, time
from tqdm import tqdm
import traceback

# width = new image width (needs to be bigger than current)
def zero_pad(width, exam_id, input_path, output_path):
    try:
        print(exam_id + ':')
        
        image = sitk.ReadImage(input_path)
        npyImage = sitk.GetArrayFromImage(image)

        if npyImage.shape[1] != npyImage.shape[2]:
            raise ValueError("Image needs to be square.")

        if width <= npyImage.shape[1]:
            raise ValueError("New width needs to be bigger than current.")    

        new_width = (width - npyImage.shape[1]) // 2
        # image_pad = np.pad(npyImage, new_width, mode='constant') # 3D
        image_pad = np.pad(npyImage, [(0, 0), (new_width, new_width), (new_width, new_width)], mode='constant') # 2D
        
        itkImage = sitk.GetImageFromArray(image_pad)
        sitk.WriteImage(itkImage, output_path)

        del image

    except Exception as e:
        print("type error: " + str(e))
        print(traceback.format_exc())
        return

def exec_zero_padding(src_dir, dst_dir, ext, width, reverse = False, desc = None):
    try:
        os.stat(dst_dir)
    except:
        os.mkdir(dst_dir)    

    input_pathAll = glob.glob(src_dir + '/*' + ext)
    input_pathAll.sort(reverse=reverse) 

    exam_ids = []
    input_paths = []
    output_paths = []

    for input_path in input_pathAll:
        exam_id = os.path.basename(input_path.replace(ext, ''))
        output_path = dst_dir + '/' + exam_id + '_zeroPedding' + ext

        # verifica se o arquivo ja existe
        if os.path.isfile(output_path):
            print('Arquivo ' + output_path + ' ja existe.')
            continue

        exam_ids.append(exam_id)
        input_paths.append(input_path)
        output_paths.append(output_path)

    for i, exam_id in enumerate(tqdm(exam_ids,desc=desc)):
        zero_pad(width, exam_id, input_paths[i], output_paths[i])
            
def main():
    ext = '.nii.gz'
    width = 640 # new width
    main_dir = '/data/flavio/anatiel/datasets/dissertacao/dataset2'
    
    src_dir = '{}'.format(main_dir)
    dst_dir = '{}/PulmoesZeroPedding'.format(main_dir)       

    exec_zero_padding(src_dir, dst_dir, ext, width, reverse = False, desc = 'Making zero padding')

if __name__=="__main__":    
    arquivo = open("anatiel/dissertacao/lungseg/time_execution_zero_padding.txt", "a")
    start = time.time()
    main()
    stop = time.time()
    exec_time = stop - start
    arquivo.write(str(exec_time))
