# GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np
import matplotlib.pyplot as plt

from data_covid import load_train_data, load_test_data
from train import unet, dice_coef

def test():
    print('Loading and preprocessing test data...')
    print('-'*30)

    imgs_test, imgs_maskt = load_test_data()
    
    #Normalization of the test set
    imgs_test = imgs_test.astype('float32')
    mean = np.mean(imgs_test)  # mean for data centering
    std = np.std(imgs_test)  # std for data normalization
    
    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std
    
    imgs_maskt = imgs_maskt.astype('float32')

    print('Loading saved weights...')
    print('-'*30)
    
    model = unet()
    model.load_weights("/home/flavio/anatiel/unet/weights_100epc.h5")

    print('Predicting masks on test data...')
    print('-'*30)
    
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save('imgs_mask_test.npy', imgs_mask_test)
    
    mask_pred = np.load('imgs_mask_test.npy')
    
    print('Saving predicted masks to files...')
    print('-' * 30)
    
    dice_test = dice_coef(imgs_maskt, mask_pred)
    print("DICE Test: ", dice_test.numpy())
    
#     plt.plot(history.history['dice_coef'])
#     plt.plot(history.history['val_dice_coef'])
#     plt.title('Model dice coeff')
#     plt.ylabel('Dice coeff')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'Test'], loc='upper left')
#     # save plot to file
#     plt.savefig('plot_dice.png')
#     plt.show()
    #plotting our dice coeff results in function of the number of epochs

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
    ax[2].set_title('Máscara GAN')

    plt.show()

if __name__ == "__main__":
    # predict
    test()
    
    # show
    #show_preds('imgs_mask_test.npy', 130)

