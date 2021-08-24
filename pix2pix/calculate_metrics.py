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
path_src_test = '/data/flavio/anatiel/datasets/dissertacao/test_images.npz'
path_mask_test = '/data/flavio/anatiel/datasets/dissertacao/test_masks.npz'
[src_images_test, tar_images_test] = load_images(path_src_test,path_mask_test)

# path_preds_pix2pix_closing_opening_best = "/data/flavio/anatiel/preds/pix2pix/best/gen2/closing_opening/"
#path_preds_pix2pix_closing_opening_best = "/data/flavio/anatiel/preds/pix2pix/best/gen2/closing_dilate/"
# path_preds_pix2pix_closing_opening_last = "/data/flavio/anatiel/preds/pix2pix/last/gen2/closing_opening/"
#path_preds_pix2pix_closing_opening_last = "/data/flavio/anatiel/preds/pix2pix/last/gen2/closing_dilate/"

# path_preds_unet_closing_opening_best = "/data/flavio/anatiel/preds/unet/best/closing_opening/"
#path_preds_unet_closing_opening_best = "/data/flavio/anatiel/preds/unet/best/closing_dilate/"
# path_preds_unet_closing_opening_last = "/data/flavio/anatiel/preds/unet/last/closing_opening/"
#path_preds_unet_closing_opening_last = "/data/flavio/anatiel/preds/unet/last/closing_dilate/"

# path_preds_unet_dice_bce = "/data/flavio/anatiel/preds/unet/dice_bce/"

w_covid_best = '/data/flavio/anatiel/models/dissertacao/unet_500epc_best.h5'
w_covid_last = '/data/flavio/anatiel/models/dissertacao/unet_500epc_last.h5'

def calculate_metrics(preds, tar_images_test):
    print('Calculating metrics...')
    print("-"*30)
    
    # morphological operations ('none', median', 'erode', 'dilate')
    #output = np.zeros((tar_images_test.shape[0], tar_images_test.shape[1], tar_images_test.shape[2]), dtype=int)
    #for i in range(tar_images_test.shape[0]):
    #    try:
    #        output[i,:,:] = morphological_operations(tar_images_test[i], op)
    #    except:
    #        output[i,:,:,0] = morphological_operations(tar_images_test[i], op)
    
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
    
    # ['', '_clahe/', '_blur/']
    pre_set = ''
    
    # ['_opening_dilate', '_closing_dilate', '_closing_opening']
    op = '_none'
    
    path = '/data/flavio/anatiel/datasets/dissertacao/gan_preds_best.npy'
    preds = np.load(path)
    
    # preds = np.load('/data/flavio/anatiel/preds/'+net+'/'+model+'/gen2/imgs_mask_test'+pre_set+op+'_gen2.npy')
    
    print(np.amin(tar_images_test))
    print(np.amax(tar_images_test))
    print(tar_images_test.dtype)
    print(np.amin(preds))
    print(np.amax(preds))
    print(preds.dtype)

    calculate_metrics(preds, tar_images_test)
    # print("="*50)
    # calculate_metrics(preds, tar_images_test, '_dilate')
    # print("="*50)
    # calculate_metrics(preds, tar_images_test, '_opening')
    # print("="*50)
    # calculate_metrics(preds, tar_images_test, '_closing')
    # print("="*50)
    # calculate_metrics(preds, tar_images_test, '_median')
    # print("="*50)