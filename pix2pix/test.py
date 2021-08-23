# GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from model import Pix2Pix
from utils import *
from losses import *
import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu

# test settings
IMG_WIDTH = 640
IMG_HEIGHT = 640
INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 1
BATCH_SIZE = 1

def test(src_images_test, path_mask_test, weights_path):
    # load model
    model = Pix2Pix(IMG_HEIGHT, IMG_WIDTH, INPUT_CHANNELS, OUTPUT_CHANNELS)
    model.compile(
        discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
        generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
        discriminator_loss = discriminator_loss,
        generator_loss = generator_loss
    )

    imgs_test, imgs_maskt = load_images(src_images_test,path_mask_test)

    # # #Normalization of the test set
    imgs_test = imgs_test.astype('float32')
    mean = np.mean(imgs_test)  # mean for data centering
    std = np.std(imgs_test)  # std for data normalization
    
    # #to float
    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std
    imgs_maskt = imgs_maskt.astype('float32')

    # predict
    print('-'*30)
    print('Loading saved weights...')
    model.load_weights(weights_path)
    print('-'*30)
    print('Predicting data...')
    output=None
    for i in range(imgs_test.shape[0]):
        pred = model.generator(imgs_test[i:i+1],training=False).numpy()
        if output is None:
            output=pred
        else:
            output = np.concatenate([output,pred],axis=0)

    # calculate metrics
    print('-'*30)
    print('Calculating metrics...')
    #print("DICE Test: ", dice(tar_images_test, output).numpy())

    print(np.amin(imgs_maskt))
    print(np.amax(imgs_maskt))
    print(imgs_maskt.dtype)

    out_max = np.amax(output)
    otsu = threshold_otsu(output)/out_max
    img_otsu = (output>otsu)*1
    img_otsu = (output<=otsu)*0
    
    print(np.amin(img_otsu))
    print(np.amax(img_otsu))
    print(img_otsu.dtype)

    
    dice, jaccard, sensitivity, specificity, accuracy, auc, prec, fscore = calc_metric(img_otsu.astype(int), imgs_maskt.astype(int))
    print("DICE: ", dice)
    print("IoU:", jaccard)
    print("Sensitivity: ", sensitivity)
    print("Specificity", specificity)
    print("ACC: ", accuracy)
    print("AUC: ", auc)
    print("Prec: ", prec)
    print("FScore: ", fscore)
    
    print('-' * 30)
    print('Saving predicted masks to files...')
    
    # remote
    # np.savez_compressed('/data/flavio/anatiel/datasets/dissertacao/test/image/teste/gan_mask_test_last_teste.npz', output)
    np.save('/data/flavio/anatiel/datasets/dissertacao/gan_preds_best.npy', output)

    # local
    # np.savez_compressed('/home/anatielsantos/mestrado/datasets/dissertacao/test/image/GanPredsLast/exam_test.npz', output)
    # np.save('/home/anatielsantos/mestrado/datasets/dissertacao/test/image/GanPredsLast/gan_preds_best.npy', output)
    
    print('-' * 30)
    
if __name__=="__main__":
    # predict remote
    w_covid_best = '/data/flavio/anatiel/models/dissertacao/gan_500epc_best.hdf5'
    w_covid_last = '/data/flavio/anatiel/models/dissertacao/gan_500epc_last.hdf5'

    # predict local
    # w_covid_best = '/home/anatielsantos/mestrado/models/dissertacao/gan/gan_500epc_best.hdf5'
    # w_covid_last = '/home/anatielsantos/mestrado/models/dissertacao/gan/gan_500epc_last.hdf5'

    # load dataset
    print('-'*30)
    print('Loading and preprocessing test data...')

    # remote
    path_src_test = '/data/flavio/anatiel/datasets/dissertacao/test_images.npz'
    path_mask_test = '/data/flavio/anatiel/datasets/dissertacao/test_masks.npz'
    # path_src_test = '/data/flavio/anatiel/datasets/dissertacao/test/image/teste/test_images_teste.npz'
    # path_mask_test = '/data/flavio/anatiel/datasets/dissertacao/test/image/teste/test_masks_teste.npz'

    # local
    # path_src_test = '/home/anatielsantos/mestrado/datasets/dissertacao/test_images.npz'
    # path_mask_test = '/home/anatielsantos/mestrado/datasets/dissertacao/test_masks.npz'
    
    test(path_src_test, path_mask_test, w_covid_best)