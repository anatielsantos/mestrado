# GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity

from data_covid import load_train_data, load_test_data, results_train
from train import unet, dice_coef
from losses import calc_metric

# w_lung = '/data/flavio/anatiel/models/new/unet2d/weights_unet_masked_lung_500epc.h5'
# w_lung_last = '/data/flavio/anatiel/models/new/unet2d/weights_unet_masked_lung_500epc_last.h5'
# w_lung_blur = '/data/flavio/anatiel/models/new/unet2d/weights_unet_masked_lung_blur_500epc.h5'
# w_lung_blur_last = '/data/flavio/anatiel/models/new/unet2d/weights_unet_masked_lung_blur_500epc_last.h5'
# w_lung_clahe = '/data/flavio/anatiel/models/new/unet2d/weights_unet_masked_lung_clahe_500epc.h5'
# w_lung_clahe_last = '/data/flavio/anatiel/models/new/unet2d/weights_unet_masked_lung_clahe_500epc_last.h5'

def test(w, imgs_test, imgs_maskt):
    print('Loading and preprocessing test data...')
    print('-'*30)
    
    #Normalization of the test set
    # imgs_test = imgs_test.astype('float32')
    # mean = np.mean(imgs_test)  # mean for data centering
    # std = np.std(imgs_test)  # std for data normalization

    imgs_test = imgs_test.astype('float32')
    imgs_test = rescale_intensity(imgs_test, in_range=(0, 1))
    
    #to float
    # imgs_test = imgs_test.astype('float32')
    # imgs_test -= mean
    # imgs_test /= std
    imgs_maskt = imgs_maskt.astype('float32')

    print('Loading saved weights...')
    print('-'*30)
    model = unet()
    model.load_weights(w)

    print('Predicting masks on test data...')
    print('-'*30)
    pred = model.predict(imgs_test, batch_size=1, verbose=1)
    
    print('Saving predicted masks to files...')
    print('-' * 30)
    np.save('/data/flavio/anatiel/datasets/dissertacao/unet_exp2_preds_best.npy', pred)
    # mask_pred = np.load('/data/flavio/anatiel/datasets/dissertacao/unet_mask_test.npy')
    
    # dice_test = dice_coef(imgs_maskt, mask_pred)
    # print("DICE Test: ", dice_test.numpy())

    # calculate metrics
    print('Calculating metrics...')
    print('-'*30)
    #print("DICE Test: ", dice(tar_images_test, output).numpy())

    print(np.amin(imgs_maskt))
    print(np.amax(imgs_maskt))
    print(imgs_maskt.dtype)
    print(np.amin(pred))
    print(np.amax(pred))
    print(pred.dtype)
    
    dice, jaccard, sensitivity, specificity, accuracy, auc, prec, fscore = calc_metric(pred.astype(int), imgs_maskt.astype(int))
    print("DICE: ", dice)
    print("IoU:", jaccard)
    print("Sensitivity: ", sensitivity)
    print("Specificity", specificity)
    print("ACC: ", accuracy)
    print("AUC: ", auc)
    print("Prec: ", prec)
    print("FScore: ", fscore)

def show_preds(path_pred, fatia):
    # load array
    data = np.load('imgs_mask_test.npy')
    
    print('Showing predicted masks (slice: ', fatia, ') ...')
    print('-' * 30)

    imgs_test, imgs_mask_test = load_test_data()
    im = imgs_test[fatia]
    im_mask = imgs_mask_test[fatia]

    fig, ax = plt.subplots(ncols=3, figsize=(12, 5))
    ax[0] = plt.subplot(1, 3, 1)
    ax[1] = plt.subplot(1, 3, 2)
    ax[2] = plt.subplot(1, 3, 3)

    ax[0].imshow(im, cmap='gray')
    ax[0].set_title('CT')

    ax[1].imshow(im_mask, cmap="gray")
    ax[1].set_title('Máscara Médico')

    ax[2].imshow(data[fatia,:,:,0], cmap="gray")
    ax[2].set_title('Máscara Unet')

    plt.show()

if __name__ == "__main__":
    # predict
    w_covid_best = '/data/flavio/anatiel/models/dissertacao/unet_exp2_200epc_best.h5'
    w_covid_last = '/data/flavio/anatiel/models/dissertacao/unet_exp2_200epc_last.h5'
    imgs_test, imgs_maskt = load_test_data()
    test(w_covid_best, imgs_test, imgs_maskt)
    # test(w_covid_last, imgs_test, imgs_maskt)
    
    # show
    # show_preds('imgs_mask_test.npy', 130)