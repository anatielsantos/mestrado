# GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from model import Pix2Pix
from utils import *
from losses import *
import numpy as np
import pandas as pd

# test settings
IMG_WIDTH = 512
IMG_HEIGHT = 512
INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 1
BATCH_SIZE = 1
w_lung_best = '/data/flavio/anatiel/models/new/pix2pix/best_weights_train_gan_512_masked_lung_500epc_gen2.hdf5'
w_lung_last = '/data/flavio/anatiel/models/new/pix2pix/best_weights_train_gan_512_masked_lung_500epc_gen2_last.hdf5'

w_lung_clahe_best = '/data/flavio/anatiel/models/new/pix2pix/best_weights_train_gan_512_masked_lung_clahe_500epc_gen2.hdf5'
w_lung_clahe_last = '/data/flavio/anatiel/models/new/pix2pix/best_weights_train_gan_512_masked_lung_clahe_500epc_gen2_last.hdf5'

w_lung_blur_best = '/data/flavio/anatiel/models/new/pix2pix/best_weights_train_gan_512_masked_lung_blur_500epc_gen2.hdf5'
w_lung_blur_last = '/data/flavio/anatiel/models/new/pix2pix/best_weights_train_gan_512_masked_lung_blur_500epc_gen2_last.hdf5'


# load dataset
print('-'*30)
print('Loading and preprocessing test data...')
path_src_test = '/data/flavio/anatiel/datasets/A/512x512/test_lung_blur.npz'
path_mask_test = '/data/flavio/anatiel/datasets/B_lesion/512x512/test.npz'
[src_images_test, tar_images_test] = load_images(path_src_test,path_mask_test)

def test(src_images_test, tar_images_test, weights_path):
    # load model
    model = Pix2Pix(IMG_HEIGHT, IMG_WIDTH, INPUT_CHANNELS, OUTPUT_CHANNELS)
    model.compile(
        discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
        generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
        discriminator_loss = discriminator_loss,
        generator_loss = generator_loss
    )

    # predict
    print('-'*30)
    print('Loading saved weights...')
    model.load_weights(weights_path)
    print('-'*30)
    print('Predicting data...')
    output=None
    for i in range(src_images_test.shape[0]):
        pred = model.generator(src_images_test[i:i+1],training=False).numpy()
        if output is None:
            output=pred
        else:
            output = np.concatenate([output,pred],axis=0)

    # calculate metrics
    print('-'*30)
    print('Calculating metrics...')
    #print("DICE Test: ", dice(tar_images_test, output).numpy())
    
    dice, jaccard, sensitivity, specificity, accuracy, auc, prec, fscore = calc_metric(output.astype(int), tar_images_test.astype(int))
    print("DICE: ", dice)
    print("IoU:", jaccard)
    print("Sensitivity: ", sensitivity)
    print("Specificity", specificity)
    print("ACC: ", accuracy)
    print("AUC: ", auc)
    print("Prec: ", prec)
    print("FScore: ", fscore)
    
    
    # print('-' * 30)
    # print('Saving predicted masks to files...')
    # np.save('imgs_mask_test.npy', output)
    # print('-' * 30)
    
if __name__=="__main__":
    # predict
    weights_path = w_lung_blur_last
    test(src_images_test, tar_images_test, weights_path)
    
    # train results
    #results = pd.read_json("/home/flavio/anatiel/pix2pix/results/new_tests/history_masked_lung_500epc_gen2.json")
    
    #print("D-Loss: ", results_train(results['d_loss']))
    #print("G-Total: ", results_train(results['g_total']))
    #print("G-Gan: ", results_train(results['g_gan']))
    #print("G-L1: ", results_train(results['g_l1']))
    #print("DICE: ", results_train(results['dice']))
    #print("Val-DICE: ", results_train(results['val_dice']))
    