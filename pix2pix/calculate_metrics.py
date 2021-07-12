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
path_src_test = '/data/flavio/anatiel/datasets/A/512x512/test.npz'
path_mask_test = '/data/flavio/anatiel/datasets/B_lesion/512x512/test.npz'
[src_images_test, tar_images_test] = load_images(path_src_test,path_mask_test)

path_preds_pix2pix_closing_opening_best = "/data/flavio/anatiel/preds/pix2pix/best/gen2/closing_opening/"
#path_preds_pix2pix_closing_opening_best = "/data/flavio/anatiel/preds/pix2pix/best/gen2/closing_dilate/"
path_preds_pix2pix_closing_opening_last = "/data/flavio/anatiel/preds/pix2pix/last/gen2/closing_opening/"
#path_preds_pix2pix_closing_opening_last = "/data/flavio/anatiel/preds/pix2pix/last/gen2/closing_dilate/"

path_preds_unet_closing_opening_best = "/data/flavio/anatiel/preds/unet/best/closing_opening/"
#path_preds_unet_closing_opening_best = "/data/flavio/anatiel/preds/unet/best/closing_dilate/"
path_preds_unet_closing_opening_last = "/data/flavio/anatiel/preds/unet/last/closing_opening/"
#path_preds_unet_closing_opening_last = "/data/flavio/anatiel/preds/unet/last/closing_dilate/"

def calculate_metrics(preds, tar_images_test, op):
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
    pre_set = '_blur'
    
    # ['_opening_dilate', '_closing_dilate', '_closing_opening']
    op = '_closing_opening'
    
    path = path_preds_unet_closing_opening_last
    preds = np.load(path+'imgs_mask_test'+pre_set+op+'.npy')
    
    # preds = np.load('/data/flavio/anatiel/preds/'+net+'/'+model+'/gen2/imgs_mask_test'+pre_set+op+'_gen2.npy')
    
    calculate_metrics(preds, tar_images_test, op)
    # print("="*50)
    # calculate_metrics(preds, tar_images_test, '_dilate')
    # print("="*50)
    # calculate_metrics(preds, tar_images_test, '_opening')
    # print("="*50)
    # calculate_metrics(preds, tar_images_test, '_closing')
    # print("="*50)
    # calculate_metrics(preds, tar_images_test, '_median')
    # print("="*50)