import os
import zipfile
import numpy as np
import tensorflow as tf
import SimpleITK as sitk
import random
import glob, time
import traceback

from tqdm import tqdm
from scipy import ndimage
from skimage.filters import threshold_otsu

@tf.function
def rotate(image, mask):
    """Rotate the volume by a few degrees"""
    
    # define some rotation angles
    angles = [-20, -10, -5, 5, 10, 20]
    # pick angles at random
    angle = random.choice(angles)

    def scipy_rotate_image(image):
        # rotate volume
        image = ndimage.rotate(image, angle, reshape=False)
        image[image < 1] = 0
        return image

    def scipy_rotate_mask(mask):
        # rotate volume
        mask = ndimage.rotate(mask, angle, reshape=False)
        otsu = threshold_otsu(mask)/np.amax(mask)
        mask[mask < otsu] = 0
        mask[mask > otsu] = 1
        return mask

    augmented_image = tf.numpy_function(scipy_rotate_image, [image], tf.float32)
    augmented_mask = tf.numpy_function(scipy_rotate_mask, [mask], tf.float32)
    return augmented_image, augmented_mask

def augmentation(exam_id, src_path, mask_path, output_path_image, output_path_mask):
    try:
        print(exam_id + ':')
        
        image = sitk.ReadImage(src_path)
        npyImage = sitk.GetArrayFromImage(image)
        npyImage = np.float32(npyImage)
        mask = sitk.ReadImage(mask_path)
        npyMask = sitk.GetArrayFromImage(mask)
        npyMask = np.float32(npyMask)

        del image
        del mask
        
        # augmentation
        print("Image Augmentation")
        volumeImage, volumeMask = rotate(np.transpose(npyImage, (1, 2, 0)), np.transpose(npyMask, (1, 2, 0)))
        volumeImage = np.transpose(volumeImage, (2, 0, 1))
        volumeMask = np.transpose(volumeMask, (2, 0, 1))

        itkImage = sitk.GetImageFromArray(volumeImage)
        sitk.WriteImage(itkImage, output_path_image)

        itkMask = sitk.GetImageFromArray(np.int16(volumeMask))
        sitk.WriteImage(itkMask, output_path_mask)

        del itkImage
        del itkMask

    except Exception as e:
        print("type error: " + str(e))
        print(traceback.format_exc())
        return
    
def exec_augmentationg(src_dir, mask_dir, dst_dir_image, dst_dir_mask, ext, reverse = False, desc = None):
    try:
        os.stat(dst_dir_image)
        os.stat(dst_dir_mask)
    except:
        os.mkdir(dst_dir_image)
        os.mkdir(dst_dir_mask)

    input_src_pathAll = glob.glob(src_dir + '/*' + ext)
    input_src_pathAll.sort(reverse=reverse) 

    input_mask_pathAll = glob.glob(mask_dir + '/*' + ext)
    input_mask_pathAll.sort(reverse=reverse) 

    exam_ids = []
    input_src_paths = []
    input_mask_paths = []
    output_paths_images = []
    output_paths_masks = []

    for input_path in input_src_pathAll:
        exam_id = os.path.basename(input_path.replace(ext, ''))
        output_path_image = dst_dir_image + '/' + exam_id + '_rotate' + ext
        output_path_mask = dst_dir_mask + '/' + exam_id + '_rotate' + ext

        # verifica se o arquivo ja existe
        if os.path.isfile(output_path_image):
            print('Arquivo ' + output_path_image + ' ja existe')
            continue
        if os.path.isfile(output_path_mask):
            print('Arquivo ' + output_path_mask + ' ja existe')
            continue

        exam_ids.append(exam_id)
        input_src_paths.append(input_path)
        output_paths_images.append(output_path_image)
        output_paths_masks.append(output_path_mask)
    
    for input_mask_path in input_mask_pathAll:
        input_mask_paths.append(input_mask_path)

    for i, exam_id in enumerate(tqdm(exam_ids,desc=desc)):
        augmentation(exam_id, input_src_paths[i], input_mask_paths[i], output_paths_images[i], output_paths_masks[i])

if __name__ == "__main__":
    dataset = 'dataset1'
    ext = '.nii.gz'
    main_dir = f'/home/anatielsantos/mestrado/datasets/dissertacao/{dataset}/image/ZeroPedding'
    main_mask_dir = f'/home/anatielsantos/mestrado/datasets/dissertacao/{dataset}/lung_mask/ZeroPedding'
    
    src_dir = '{}'.format(main_dir)
    mask_dir = '{}'.format(main_mask_dir)

    dst_dir_image = '{}/data_augmentation'.format(main_dir)
    dst_dir_mask = '{}/data_augmentation'.format(main_mask_dir)

    exec_augmentationg(src_dir, mask_dir, dst_dir_image, dst_dir_mask, ext, reverse = False, desc = f'Augmentation from {dataset}')