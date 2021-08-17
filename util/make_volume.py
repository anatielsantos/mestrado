# GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import SimpleITK as sitk
import numpy as np
import glob

def load_patient(image):
    itkImage = sitk.ReadImage(image)
    npyImage = sitk.GetArrayFromImage(itkImage)
    npyImage = np.expand_dims(npyImage, axis=-1)

    return npyImage

# load dataset
def load_pred(path_pred):
    tar_npz = np.load(path_pred, allow_pickle=True)
    tar = tar_npz['arr_0']
    
    return np.float32(tar)

def make_volume(src_dir, path_pred):
    for input_path in src_dir:
        src = load_patient(input_path)
        pred = load_pred(path_pred)

        print("src shape: ", src.shape)
        print("pred shape: ", pred.shape)

def main():
    src_path = '/data/flavio/anatiel/datasets/dissertacao/test/image/*.nii.gz'
    output_path = '/data/flavio/anatiel/datasets/dissertacao'

    input_pathAll = glob.glob(src_path)
    input_pathAll.sort(reverse=False)

    # load dataset
    print('-'*30)
    print('Loading and preprocessing test data...')
    path_image_test = '/data/flavio/anatiel/datasets/dissertacao/test_images.npz'
    path_mask_test = '/data/flavio/anatiel/datasets/dissertacao/gan_mask_test.npz'
    # [src_images_test, tar_images_test] = load_pred(path_mask_test)

    make_volume(input_pathAll, path_mask_test)

    # itkImage = sitk.GetImageFromArray(tar_images_test)
    # sitk.WriteImage(itkImage, output_path)