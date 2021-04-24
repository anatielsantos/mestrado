import numpy as np

def load_train_data():
    path_src_train = np.load('/home/anatielsantos/workspace_visual/mestrado/datasets/covid19/A/512x512/masked/train_masked.npz')
    path_mask_train = np.load('/home/anatielsantos/workspace_visual/mestrado/datasets/covid19/B-lung/512x512/masked_lung/train_masked_lung.npz')
    imgs_train = path_src_train['arr_0']
    masks_train = path_mask_train['arr_0']

    return imgs_train, masks_train

def load_test_data():
    imgs_test_npz = np.load('/home/anatielsantos/workspace_visual/mestrado/datasets/covid19/A/512x512/masked/test_masked.npz')
    masks_test_npz = np.load('/home/anatielsantos/workspace_visual/mestrado/datasets/covid19/B-lung/512x512/masked_lung/test_masked_lung.npz')
    imgs_test = imgs_test_npz['arr_0']
    masks_test = masks_test_npz['arr_0']

    return imgs_test, masks_test

def load_val_data():
    imgs_val_npz = np.load('/home/anatielsantos/workspace_visual/mestrado/datasets/covid19/A/512x512/masked/val_masked.npz')
    masks_val_npz = np.load('/home/anatielsantos/workspace_visual/mestrado/datasets/covid19/B-lung/512x512/masked_lung/val_masked_lung.npz')
    imgs_val = imgs_val_npz['arr_0']
    masks_val = masks_val_npz['arr_0']

    return imgs_val, masks_val