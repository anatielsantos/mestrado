# GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# import SimpleITK as sitk
import numpy as np

output_path = '/data/flavio/anatiel/datasets/dissertacao'

# load dataset
def load_images(path_pred, path_image):
    src_npz = np.load(path_image)
    tar_npz = np.load(path_pred)
    src = src_npz['arr_0']
    tar = tar_npz['arr_0']
    
    return [src, tar]

# load dataset
print('-'*30)
print('Loading and preprocessing test data...')
path_image_test = '/data/flavio/anatiel/datasets/dissertacao/test_images.npz'
path_mask_test = '/data/flavio/anatiel/datasets/dissertacao/gan_mask_test.npz'
[src_images_test, tar_images_test] = load_images(path_image_test, path_mask_test)

print(src_images_test.shape)
print(tar_images_test.shape)

# itkImage = sitk.GetImageFromArray(tar_images_test)
# sitk.WriteImage(itkImage, output_path)