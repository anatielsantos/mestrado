# GPU
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import Pix2Pix
from tensorflow.keras.callbacks import ModelCheckpoint
from utils import *
from losses import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# configuração necessária nas GPU's RTX
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

# train settings
BUFFER_SIZE = 400
BATCH_SIZE = 1
EPOCHS = 150
IMG_WIDTH = 640
IMG_HEIGHT = 640
INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 1


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

    src = np.float32(np.expand_dims(np.concatenate(src), axis=-1))
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

    # Normalization of the train set (Exp 2)
    # src_images_train = src_images_train.astype('float32')
    # src_images_train = rescale_intensity(src_images_train, in_range=(-1, 1))

    # train model
    checkpoint = ModelCheckpoint(
        path_weights+'gan_test2_exp1_1_best.hdf5',
        monitor='dice',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='max'
    )
    # checkpoint2 = ModelCheckpoint(
    #     path_weights+'best_weights_val__test2_512_masked_lung_blur_500epc.hdf5',
    #     monitor='val_dice',
    #     verbose=1,
    #     save_best_only=True,
    #     save_weights_only=True,
    #     mode='max'
    # )

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
    # history=model.fit(
    #     src_images_train,
    #     tar_images_train,
    #     batch_size=BATCH_SIZE,
    #     epochs=EPOCHS,
    #     callbacks=[checkpoint,checkpoint2],
    #     validation_data=(src_images_val, tar_images_val)
    # )

    model.save(path_weights+'gan_test2_exp1_1_last.hdf5')

    # convert the history.history dict to a pandas DataFrame:
    hist_df = pd.DataFrame(history.history)

    # save to json:
    print("Saving history")
    hist_json_file = path_json+'gan_test2_exp1_1_history_150epc.json'
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
    plt.savefig(path_plot+'gan_test2_exp1_1_plot_150epc.png')
    # plt.show()


if __name__ == "__main__":
    # dataset path remote
    path_src_train = """
        /data/flavio/anatiel/datasets/dissertacao/
        final_tests/train_images_exp1_lesion.npz
    """
    path_mask_train = """
        /data/flavio/anatiel/datasets/dissertacao
        /final_tests/train_masks_exp1_lesion.npz
    """

    # dataset path local
    # path_src_train = """
    #   /home/anatielsantos/mestrado/datasets/
    #   dissertacao/train_images.npz
    # """
    # path_mask_train = """
    #   /home/anatielsantos/mestrado/datasets/
    #   dissertacao/train_masks.npz
    # """

    # paths save
    path_weights = '/data/flavio/anatiel/models/dissertacao/final_tests/'
    path_json = '/data/flavio/anatiel/models/dissertacao/final_tests/'
    path_plot = '/data/flavio/anatiel/models/dissertacao/final_tests/'

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

    # model training
    train(path_weights, src_images_train, tar_images_train)
