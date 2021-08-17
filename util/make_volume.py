# GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# import SimpleITK as sitk
import numpy as np

output_path = '/data/flavio/anatiel/datasets/dissertacao'

# load dataset
def load_images(path_mask):
    tar_npz = np.load(path_mask)
    tar = tar_npz['arr_0']
    
    return tar

# load dataset
print('-'*30)
print('Loading and preprocessing test data...')
path_mask_test = '/data/flavio/anatiel/datasets/dissertacao/gan_mask_test.npy'
tar_images_test = load_images(path_mask_test)

print(tar_images_test.shape)

# itkImage = sitk.GetImageFromArray(tar_images_test)
# sitk.WriteImage(itkImage, output_path)