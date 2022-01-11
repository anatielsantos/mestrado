import numpy as np
from glob import glob
from sklearn.model_selection import KFold
# from util.compress_dataset import compress_dataset


ext = '.nii.gz'
dataset = 'dataset1'  # [dataset1, dataset2]
joint = 'train'  # [train, test]
main_dir_image = f'/home/anatielsantos/mestrado/datasets/dissertacao/bbox/{dataset}/images/'
main_dir_mask = f'/home/anatielsantos/mestrado/datasets/dissertacao/bbox/{dataset}/masks/'

src = main_dir_image
tar = main_dir_mask
src_dir = '{}'.format(src)
mask_dir = '{}'.format(tar)

input_src_pathAll = glob(src_dir + '/*' + ext)
input_src_pathAll.sort(reverse=False)

input_mask_pathAll = glob(main_dir_mask + '/*' + ext)
input_mask_pathAll.sort(reverse=False)

kf = KFold(n_splits=10)
kf.get_n_splits(input_src_pathAll)

X_train, X_test, y_train, y_test = list(), list(), list(), list()

for train_index, test_index in kf.split(input_src_pathAll):
    print("TRAIN:", train_index, "TEST:", test_index)

print(train_index[1])


def make_folds():
    pass


# if __name__ == '__main__':
    # data = compress_dataset()

