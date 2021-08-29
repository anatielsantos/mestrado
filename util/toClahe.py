# GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import SimpleITK as sitk
import numpy as np
import glob
from skimage.exposure import equalize_adapthist
from skimage.exposure import rescale_intensity
from tqdm import tqdm

# clahe equalization
def equalize_clahe(images, npyMask):
    print("Clahe equalization...")
    final_img = np.int16(images)
    for i in range(images.shape[0]):
        imgClahe = images[i][:,:,0] / np.amax(images[i][:,:,0])
        imgClahe = equalize_adapthist(imgClahe, clip_limit=0.02)
        final_img[i][:,:,0] = imgClahe * np.amax(images[i][:,:,0])
    
    
    new_final_img = final_img * npyMask

    # final_img = images[35][:,:,0] / np.amax(images[35][:,:,0])
    # final_img = equalize_adapthist(final_img, clip_limit=0.02)
    # final_img = np.int16(final_img * np.amax(images[35][:,:,0]))
                                 
    return new_final_img

def saveClahe(exam_id, input_path, mask_path, output_path):
    print(exam_id + ':')
    itkImage = sitk.ReadImage(input_path)
    npyImage = sitk.GetArrayFromImage(itkImage)
    npyImage = np.expand_dims(npyImage, axis=-1)

    mask = sitk.ReadImage(mask_path)
    npyMask = sitk.GetArrayFromImage(mask)
    npyMask = np.expand_dims(npyMask, axis=-1)
    
    npyImageClahe = equalize_clahe(npyImage, npyMask)

    itkImage = sitk.GetImageFromArray(npyImageClahe)

    sitk.WriteImage(itkImage, output_path)

    del itkImage

def execSaveClahe(src_dir, mask_dir, dst_dir, ext, search_pattern, reverse = False, desc = None):
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
        output_path = dst_dir + '/' + exam_id + '_clahe' + ext

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

    for i, exam_id in enumerate(tqdm(exam_ids,desc=desc)):
        saveClahe(exam_id, input_paths[i], input_mask_paths[i], output_paths[i])

def main():

    ext = '.nii.gz'
    search_pattern = '*'
    dataset = 'test'

    # local
    main_dir = '/home/anatielsantos/mestrado/datasets/dissertacao/dataset1/image/ZeroPedding/lung_extracted'
    main_mask_dir = '/home/anatielsantos/mestrado/datasets/dissertacao/dataset1/image/ZeroPedding/PulmoesMascara/PulmoesMascaraFillHoles'

    # remote
    # main_dir = f'/data/flavio/anatiel/datasets/dissertacao/{dataset}/image'
    # model_path = '/data/flavio/anatiel/models/dissertacao/unet_500epc_last.h5'

    src_dir = '{}'.format(main_dir)
    dst_dir = '{}/Clahe'.format(main_dir)

    mask_dir = '{}'.format(main_mask_dir)

    execSaveClahe(src_dir, mask_dir, dst_dir, ext, search_pattern, reverse = False, desc = 'To Clahe')

if __name__ == '__main__':
    main()