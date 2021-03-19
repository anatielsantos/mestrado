# GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from model import Pix2Pix
from utils import *
from losses import *
import numpy as np
from pix2pix2 import load_images

# test settings
IMG_WIDTH = 256
IMG_HEIGHT = 256
INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 1
BATCH_SIZE = 1
weights_path = '/data/flavio/anatiel/models/best_weights_train_gan.hdf5'

# load dataset
print('-'*30)
print('Loading and preprocessing test data...')
path_src_test = '/data/flavio/anatiel/datasets/A/test.npz'
path_mask_test = '/data/flavio/anatiel/datasets/B_lesion/test.npz'
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
    output=None
    for i in range(src_images_test.shape[0]):
        pred = model.generator(src_images_test[i:i+1],training=False).numpy()
        if output is None:
            output=pred
        else:
            output = np.concatenate([output,pred],axis=0)

    # generate image from source
    # print("GEN DIRETO2",model.generator(src_images_test[0:3],training = False).numpy())
    # src_images_test,mask = src_images_test[0:3],mask[0:3]
    # gen_image = model.predict(src_images_test,batch_size=BATCH_SIZE)
    # print("GEN INDIRETO",gen_image)

    # calculate metrics
    print('-' * 30)
    print("DICE Test: ", dice(tar_images_test, output).numpy())

    print('-' * 30)
    print('Saving predicted masks to files...')
    np.save('imgs_mask_test.npy', output)
    print('-' * 30)
    
if __name__=="__main__":
    # predict
    test(src_images_test, tar_images_test, weights_path)