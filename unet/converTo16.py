# GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import SimpleITK as sitk
import numpy as np
import glob
from tqdm import tqdm

def save16(exam_id, input_path, output_path):
    print(exam_id + ':')
    itkImage = sitk.ReadImage(input_path)
    npyImage = sitk.GetArrayFromImage(itkImage)
    npyImage = np.expand_dims(npyImage, axis=-1)
    npyImage16 = np.int16(npyImage)

    itkImage = sitk.GetImageFromArray(npyImage16)

    sitk.WriteImage(itkImage, output_path)

    del itkImage

def execSave16(src_dir, dst_dir, ext, search_pattern, reverse = False, desc = None):
    try:
        os.stat(dst_dir)
    except:
        os.mkdir(dst_dir)

    input_pathAll = glob.glob(src_dir + '/' + search_pattern + ext)
    input_pathAll.sort(reverse=reverse)

    exam_ids = []
    input_paths = []
    output_paths = []

    for input_path in input_pathAll:
        exam_id = os.path.basename(input_path.replace(ext, ''))
        output_path = dst_dir + '/' + exam_id + '_16' + ext

        # verifica se o arquivo ja existe
        if os.path.isfile(output_path):
            print('Arquivo ' + output_path + ' ja existe')
            # os.remove(output_path)
            continue

        exam_ids.append(exam_id)
        input_paths.append(input_path)
        output_paths.append(output_path)

        for i, exam_id in enumerate(tqdm(exam_ids,desc=desc)):
            save16(exam_id, input_paths[i], output_paths[i])

def main():

    ext = '.nii.gz'
    search_pattern = '*'
    dataset = 'test'

    # local
    main_dir = '/home/anatielsantos/mestrado/datasets/dissertacao/dataset1/image/ZeroPedding/lung_extracted'

    # remote
    # main_dir = f'/data/flavio/anatiel/datasets/dissertacao/{dataset}/image'
    # model_path = '/data/flavio/anatiel/models/dissertacao/unet_500epc_last.h5'

    src_dir = '{}'.format(main_dir)
    dst_dir = '{}/Image16'.format(main_dir)

    execSave16(src_dir, dst_dir, ext, search_pattern, reverse = False, desc = 'To 16 bits')

if __name__ == '__main__':
    main()