# GPU
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import Pix2Pix
from tensorflow.keras.callbacks import ModelCheckpoint
# from util import *
from losses import *


# configuração necessária nas GPU's RTX
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.InteractiveSession(config=config)

# train settings
BUFFER_SIZE = 400
BATCH_SIZE = 1
EPOCHS = 150
IMG_WIDTH = 544
IMG_HEIGHT = 544
INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 1
KF = "9"  # Definir o fold
GPU = "4"  # Definir a GPU
DS = "_mixed_quant"  # Definir o dataset
 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

# loss
def generator_loss(disc_generated_output, gen_output, target, LAMBDA=100):
    gan_loss = loss_object(
        tf.ones_like(disc_generated_output),
        disc_generated_output
    )

    # mean absolute error
    # l1_loss = tf.reduce_mean(tf.abs(target - gen_output)) # default

    # 1 loss function
    l1_loss = dice_loss(target, gen_output)

    # 2 loss function
    # l1_loss = dice_bce_loss(target, gen_output)

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


# load dataset
def load_images(path_src, path_mask):
    src_npz = np.load(path_src, allow_pickle=True)
    tar_npz = np.load(path_mask, allow_pickle=True)
    src = src_npz['arr_0']
    tar = tar_npz['arr_0']

    src = np.expand_dims(np.concatenate(src), axis=-1)
    tar = np.expand_dims(np.concatenate(tar), axis=-1)

    return src, tar


def train(path_weights, src_images_train, tar_images_train):
    # dataset = [src_images_train, tar_images_train]

    # createing pix2pix
    model = Pix2Pix(IMG_HEIGHT, IMG_WIDTH, INPUT_CHANNELS, OUTPUT_CHANNELS)
    model.compile(
        discriminator_optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
        generator_optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
        discriminator_loss=discriminator_loss,
        generator_loss=generator_loss,
        metrics=['accuracy', dice]
    )

    # train model
    checkpoint = ModelCheckpoint(
        path_weights+'gan_ds'+DS+'_'+KF+'_150epc_best.hdf5',
        monitor='dice',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='max'
    )

    history = model.fit(
        src_images_train,
        tar_images_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        shuffle=True,
        validation_split=0.1,
        callbacks=[checkpoint]
    )

    model.save(path_weights+'gan_ds'+DS+'_'+KF+'_150epc_last.hdf5')

    # convert the history.history dict to a pandas DataFrame:
    hist_df = pd.DataFrame(history.history)

    # save to json:
    print("Saving history")
    hist_json_file = path_json+'gan_ds'+DS+'_'+KF+'_150epc.json'
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)
    print("History saved")

    plt.plot(history.history['dice'])
    plt.plot(history.history['val_dice'])
    plt.plot(history.history['g_l1'])
    plt.title('Model dice coeff')
    plt.ylabel('Dice coeff')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val', 'Loss'], loc='upper left')
    # save plot to file
    plt.savefig(path_plot+'gan_ds'+DS+'_'+KF+'_150epc.png')
    # plt.show()


if __name__ == "__main__":
    # dataset path
    path_src_train = f"/data/flavio/anatiel/datasets/dissertacao/final_tests/kfold/dataset_mixed/quant/images{DS}_k{KF}.npz"

    path_mask_train = f"/data/flavio/anatiel/datasets/dissertacao/final_tests/kfold/dataset_mixed/quant/masks{DS}_k{KF}.npz"

    # paths save
    path_weights = f'/data/flavio/anatiel/models/models_ds{DS}/'
    path_json = f'/data/flavio/anatiel/models/models_ds{DS}/'
    path_plot = f'/data/flavio/anatiel/models/models_ds{DS}/'

    # load dataset
    [src_images_train, tar_images_train] = load_images(
        path_src_train,
        path_mask_train
    )

    print(
        'Loaded train images: ',
        src_images_train.shape,
        tar_images_train.shape
    )

    print(f"Running K{KF} from DATASET {DS} on GPU {GPU}")

    # Normalization of the train set (Exp 1)
    # mean = np.mean(src_images_train)  # mean for data centering
    # std = np.std(src_images_train)  # std for data normalization
    # src_images_train = src_images_train.astype(np.float32)
    # src_images_train -= mean
    # src_images_train /= std

    # src_images_train = src_images_train.astype(np.float32)
    # tar_images_train = tar_images_train.astype(np.float32)

    # model training
    train(path_weights, src_images_train, tar_images_train)
