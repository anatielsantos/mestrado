# GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from utils import *
from losses import *
import numpy as np

# load dataset
print('Loading test data...')
print("-"*30)

# remote
path_src_test = '/data/flavio/anatiel/datasets/dissertacao/test_images.npz'
path_mask_test = '/data/flavio/anatiel/datasets/dissertacao/test_masks.npz'
[src_images_test, tar_images_test] = load_images(path_src_test,path_mask_test)

# remote
w_covid_best = '/data/flavio/anatiel/models/dissertacao/unet_500epc_best.h5'
w_covid_last = '/data/flavio/anatiel/models/dissertacao/unet_500epc_last.h5'


def calculate_metrics(preds, tar_images_test):
    print('Calculating metrics...')
    print("-"*30)

    dice, jaccard, sensitivity, specificity, accuracy, auc, prec, fscore = calc_metric(preds.astype(int), tar_images_test.astype(int))

    print("DICE: ", dice)
    print("IoU:", jaccard)
    print("Sensitivity: ", sensitivity)
    print("Specificity", specificity)
    print("ACC: ", accuracy)
    print("AUC: ", auc)
    print("Prec: ", prec)
    print("FScore: ", fscore)


if __name__=="__main__":
    
    path = '/data/flavio/anatiel/datasets/dissertacao/gan_preds_best.npy'
    preds = np.load(path)

    print(np.amin(tar_images_test))
    print(np.amax(tar_images_test))
    print(tar_images_test.dtype)
    print(np.amin(preds))
    print(np.amax(preds))
    print(preds.dtype)

    calculate_metrics(preds, tar_images_test)
