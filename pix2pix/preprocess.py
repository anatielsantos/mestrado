import numpy as np
from utils import *

# GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# dataset path
path_save = "/data/flavio/anatiel/datasets/A/512x512"

# path_src_train = '/data/flavio/anatiel/datasets/A/512x512/train_masked.npz'
# path_mask_train = '/data/flavio/anatiel/datasets/B_lesion/512x512/train_masked.npz'

# path_src_val = '/data/flavio/anatiel/datasets/A/512x512/val_masked.npz'
# path_mask_val = '/data/flavio/anatiel/datasets/B_lesion/512x512/val_masked.npz'

# path_src_test_masked = '/data/flavio/anatiel/datasets/A/512x512/test_masked.npz'
# path_mask_test_masked = '/data/flavio/anatiel/datasets/B_lesion/512x512/test_masked.npz'

path_src_train = '/data/flavio/anatiel/datasets/A/512x512/train_lung.npz'
path_mask_train = '/data/flavio/anatiel/datasets/B_lung/512x512/train_lung.npz'

path_src_val = '/data/flavio/anatiel/datasets/A/512x512/val_lung.npz'
path_mask_val = '/data/flavio/anatiel/datasets/B_lung/512x512/val_lung.npz'

path_src_test = '/data/flavio/anatiel/datasets/A/512x512/test_lung.npz'
path_mask_test = '/data/flavio/anatiel/datasets/B_lung/512x512/test_lung.npz'

# load dataset
print("Loading dataset...")
[src_images_train, tar_images_train] = load_images(path_src_train, path_mask_train)
[src_images_val, tar_images_val] = load_images(path_src_val, path_mask_val)
[src_images_test, tar_images_test] = load_images(path_src_test, path_mask_test)

# preprocessing
print("Preprocessing train...")
# img_norm = imadjust(src_images_train,src_images_train.min(),src_images_train.max(),0,1)
#img_blur = blur_image(src_images_train)
# final_img = extract_lung(src_images_train, tar_images_train)
# clahe_img =  equalize_clahe(img_norm_lung)
final_image = bg_blck(src_images_train)
print("Saving train data...")
np.savez_compressed(f"{path_save}/train_bgblack",final_img)

print("Preprocessing val...")
# img_norm = imadjust(src_images_val,src_images_val.min(),src_images_val.max(),0,1)
# img_norm_lung = extract_lung(img_norm, src_images_val)
# clahe_img =  equalize_clahe(img_norm_lung)
# final_image = blur_image(src_images_val)
# print("Saving val data")
# np.savez_compressed(f"{path_save}/val_masked_lung_blur",final_img)

print("Preprocessing test...")
# img_norm = imadjust(src_images_test,src_images_test.min(),src_images_test.max(),0,1)
# img_norm_lung = extract_lung(img_norm, src_images_test)
# final_img = equalize_clahe(img_norm_lung)
# final_image = blur_image(src_images_test)
# print("Saving test data")
# np.savez_compressed(f"{path_save}/test_masked_lung_blur",final_img)