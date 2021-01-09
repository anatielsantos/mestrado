import logging
logging.getLogger('tensorflow').disabled = True

# Import TensorFlow >= 1.10 and enable eager execution
from tensorflow.python.client import device_lib
from matplotlib import pyplot as plt
import cv2
import glob
from IPython.display import clear_output
import PIL
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import tensorflow as tf
import matplotlib.image as mpimg

# %matplotlib inline
# print(device_lib.list_local_devices())
# tf.__version__

BUFFER_SIZE = 2000
BATCH_SIZE = 1
IMG_WIDTH = 512
IMG_HEIGHT = 512


def Prepare_dataset(Dataset_dir):

    # SB_dir = os.path.join(Dataset_dir, 'Croped_SB/')  # Satellite Bilder path
    # GT_dir = os.path.join(Dataset_dir, 'Croped_GT/')  # Ground Truths path
    SB_dir = os.path.join(Dataset_dir, 'A/')  # image CT
    GT_dir = os.path.join(Dataset_dir, 'B/')  # mask

    # SB_listnames = glob.glob(SB_dir+"*.png")  # Satellite Bilder filenames
    # GT_listnames = glob.glob(GT_dir+"*.png")  # Ground Truths filenames
    SB_listnames = glob.glob(SB_dir+"*.jpg")  # image CT filenames
    GT_listnames = glob.glob(GT_dir+"*.jpg")  # mask filenames

    GT_listnames.sort()
    SB_listnames.sort()

    # print("Satellite Directory:", SB_dir)
    # print('Anzahl der  Ground Truths:', len(SB_listnames))
    # print("")
    # print("Ground Truths Directory:", GT_dir)
    # print('Anzahl der satellien Bilder:', len(GT_listnames))
    print("Images Dir:", SB_dir)
    print('CT Images:', len(SB_listnames))
    print("-----------")
    print("Mask Dir:", GT_dir)
    print('Masks:', len(GT_listnames))

    print("=================================================")

    return SB_dir, GT_dir, SB_listnames, GT_listnames


def Extract_Contour(image):

    # Load an color image in grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    kernel = np.ones((2, 2), np.uint8)
    erosion = cv2.erode(gray, kernel, iterations=2)

    blur = cv2.GaussianBlur(erosion, (5, 5), 0)
    (t, maskLayer) = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((2, 2), np.uint8)
    dilation = cv2.dilate(maskLayer, kernel, iterations=1)

    blur1 = cv2.GaussianBlur(dilation, (5, 5), 0)
    (t, binary) = cv2.threshold(blur1, 0, 1,  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    (contours, _) = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)
    mask = np.zeros(image.shape, dtype="uint8")
    cv2.drawContours(mask, contours, -1, (0, 255, 0), 7)

    return mask


def Contour_overlay_GT(GT_img, Contour_array):

    # ContourOverGT = cv2.bitwise_or(GT_img, Contour_array)
    ContourOverGT = cv2.addWeighted(GT_img, 1, Contour_array, 1, 0)
    # ContourOverSB = cv2.bitwise_or(GT_img, )

    ContourOverGT = cv2.GaussianBlur(ContourOverGT, (5, 5), 0)

    return ContourOverGT


def load_image(SB_path, GT_path, is_train, return_name):

    SB_path = str(SB_path).split("'")[1]
    SB_img = cv2.imread(SB_path)
    SB_img = cv2.cvtColor(SB_img, cv2.COLOR_BGR2RGB)

    GT_path = str(GT_path).split("'")[1]
    GT_img = cv2.imread(GT_path)
    GT_img = cv2.cvtColor(GT_img, cv2.COLOR_BGR2RGB)

    Contour = Extract_Contour(GT_img)
    ContourOverGT = Contour_overlay_GT(GT_img, Contour)

    if is_train:
        if np.random.random() > 0.5:
         # random mirroring
            SB_img = cv2.flip(SB_img, -1)
            ContourOverGT = cv2.flip(ContourOverGT, -1)

    SB_img = (SB_img / 127.5) - 1
    ContourOverGT = (ContourOverGT / 127.5) - 1

    SB_img = SB_img.astype("float32")
    ContourOverGT = ContourOverGT.astype("float32")

    if return_name:
        return SB_img, ContourOverGT, GT_path

    return SB_img, ContourOverGT


# Train_Dataset_dir = "/media/immopixel/Amir/server/dataset/tirol/Klein_Dataset2/train/"
# Valid_Dataset_dir = "/media/immopixel/Amir/server/dataset/tirol/Klein_Dataset2/valid/"
# Test_Dataset_dir = "/media/immopixel/Amir/server/dataset/tirol/Klein_Dataset2/test/"
Train_Dataset_dir = "../datasets/covid19_2/train/"
Valid_Dataset_dir = "../datasets/covid19_2/val/"
Test_Dataset_dir = "../datasets/covid19_2/test/"

# Diretório atual da rede
Dir_atual = "../pix2pix-a2amir/"

Train_SB_dir, Train_GT_dir, SB_Train_listnames, GT_Train_listnames = Prepare_dataset(
    Train_Dataset_dir)  # load Satellite und Ground Truths Data
Valid_SB_dir, Valid_GT_dir, SB_Valid_listnames, GT_Valid_listnames = Prepare_dataset(
    Valid_Dataset_dir)
Test_SB_dir, Test_GT_dir, SB_Test_listnames, GT_Test_listnames = Prepare_dataset(
    Test_Dataset_dir)

# train_dataset = tf.data.Dataset.from_tensor_slices(((SB_filenames, GT_filenames)))
SB_dataset_train = tf.data.Dataset.from_tensor_slices(SB_Train_listnames)
GT_dataset_train = tf.data.Dataset.from_tensor_slices(GT_Train_listnames)
train_dataset = tf.data.Dataset.zip((SB_dataset_train, GT_dataset_train))
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.map(lambda x, y: tf.py_function(
    load_image, [x, y, True, False], [tf.float32, tf.float32]), num_parallel_calls=4)
train_dataset = train_dataset.batch(1)

SB_dataset_valid = tf.data.Dataset.from_tensor_slices(SB_Valid_listnames)
GT_dataset_valid = tf.data.Dataset.from_tensor_slices(GT_Valid_listnames)
valid_dataset = tf.data.Dataset.zip((SB_dataset_valid, GT_dataset_valid))
valid_dataset = valid_dataset.shuffle(BUFFER_SIZE)

valid_dataset = valid_dataset.map(lambda x, y: tf.py_function(
    load_image, [x, y, True, False], [tf.float32, tf.float32]), num_parallel_calls=4)
valid_dataset = valid_dataset.batch(1)

SB_dataset_test = tf.data.Dataset.from_tensor_slices(SB_Test_listnames)
GT_dataset_test = tf.data.Dataset.from_tensor_slices(GT_Test_listnames)
test_dataset = tf.data.Dataset.zip((SB_dataset_test, GT_dataset_test))
test_dataset = test_dataset.shuffle(BUFFER_SIZE)

test_dataset = test_dataset.map(lambda x, y: tf.py_function(load_image, [
                                x, y, False, True], [tf.float32, tf.float32, tf.string]), num_parallel_calls=4)
test_dataset = test_dataset.batch(1)

# for SB, ContourOverGT in train_dataset.take(10):
#    print(name)
#    plt.figure(figsize=(15,15))
#    display_list=[SB[0,:,:,:],ContourOverGT[0,:,:,:]]
#    title = ['Input Image', 'ContourOverGT Image']
#    for i in range(2):
#        plt.subplot(1, 2, i+1)
#        plt.title(title[i])
#        plt.imshow(display_list[i]*.5+.5 )
#        plt.axis('off')
#    plt.show()


# MODEL GENERATE
OUTPUT_CHANNELS = 3


class Downsample(tf.keras.Model):
    def __init__(self, filters, size, apply_batchnorm=True):
        super(Downsample, self).__init__()
        self.apply_batchnorm = apply_batchnorm
        initializer = tf.random_normal_initializer(0., 0.02)

        self.conv1 = tf.keras.layers.Conv2D(filters,
                                            (size, size),
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False)
        if self.apply_batchnorm:
            self.batchnorm = tf.keras.layers.BatchNormalization()

    def call(self, x, training):
        x = self.conv1(x)
        if self.apply_batchnorm:
            x = self.batchnorm(x, training=training)
        x = tf.nn.leaky_relu(x)
        return x


class Upsample(tf.keras.Model):
    def __init__(self, filters, size, apply_dropout=False):
        super(Upsample, self).__init__()
        self.apply_dropout = apply_dropout
        initializer = tf.random_normal_initializer(0., 0.02)

        self.up_conv = tf.keras.layers.Conv2DTranspose(filters,
                                                       (size, size),
                                                       strides=2,
                                                       padding='same',
                                                       kernel_initializer=initializer,
                                                       use_bias=False)
        self.batchnorm = tf.keras.layers.BatchNormalization()
        if self.apply_dropout:
            self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, x1, x2, training):
        x = self.up_conv(x1)
        x = self.batchnorm(x, training=training)
        if self.apply_dropout:
            x = self.dropout(x, training=training)
        x = tf.nn.relu(x)
        x = tf.concat([x, x2], axis=-1)
        return x


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)

        self.down1 = Downsample(64, 4, apply_batchnorm=False)
        self.down2 = Downsample(128, 4)
        self.down3 = Downsample(256, 4)
        self.down4 = Downsample(512, 4)
        self.down5 = Downsample(512, 4)
        self.down6 = Downsample(512, 4)
        self.down7 = Downsample(512, 4)
        self.down8 = Downsample(512, 4)

        self.up1 = Upsample(512, 4, apply_dropout=True)
        self.up2 = Upsample(512, 4, apply_dropout=True)
        self.up3 = Upsample(512, 4, apply_dropout=True)
        self.up4 = Upsample(512, 4)
        self.up5 = Upsample(256, 4)
        self.up6 = Upsample(128, 4)
        self.up7 = Upsample(64, 4)

        self.last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS,
                                                    (4, 4),
                                                    strides=2,
                                                    padding='same',
                                                    kernel_initializer=initializer)

   # @tf.contrib.eager.defun
    def call(self, x, training):
        # x shape == (bs, 256, 256, 3)
        x1 = self.down1(x, training=training)  # (bs, 128, 128, 64)
        x2 = self.down2(x1, training=training)  # (bs, 64, 64, 128)
        x3 = self.down3(x2, training=training)  # (bs, 32, 32, 256)
        x4 = self.down4(x3, training=training)  # (bs, 16, 16, 512)
        x5 = self.down5(x4, training=training)  # (bs, 8, 8, 512)
        x6 = self.down6(x5, training=training)  # (bs, 4, 4, 512)
        x7 = self.down7(x6, training=training)  # (bs, 2, 2, 512)
        x8 = self.down8(x7, training=training)  # (bs, 1, 1, 512)

        x9 = self.up1(x8, x7, training=training)  # (bs, 2, 2, 1024)
        x10 = self.up2(x9, x6, training=training)  # (bs, 4, 4, 1024)
        x11 = self.up3(x10, x5, training=training)  # (bs, 8, 8, 1024)
        x12 = self.up4(x11, x4, training=training)  # (bs, 16, 16, 1024)
        x13 = self.up5(x12, x3, training=training)  # (bs, 32, 32, 512)
        x14 = self.up6(x13, x2, training=training)  # (bs, 64, 64, 256)
        x15 = self.up7(x14, x1, training=training)  # (bs, 128, 128, 128)

        x16 = self.last(x15)  # (bs, 256, 256, 3)
        x16 = tf.nn.tanh(x16)

        return x16

# MODEL DISCRIMINATOR


class DiscDownsample(tf.keras.Model):
    def __init__(self, filters, size, apply_batchnorm=True):
        super(DiscDownsample, self).__init__()
        self.apply_batchnorm = apply_batchnorm
        initializer = tf.random_normal_initializer(0., 0.02)

        self.conv1 = tf.keras.layers.Conv2D(filters,
                                            (size, size),
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False)
        if self.apply_batchnorm:
            self.batchnorm = tf.keras.layers.BatchNormalization()

    def call(self, x, training):
        x = self.conv1(x)
        if self.apply_batchnorm:
            x = self.batchnorm(x, training=training)
        x = tf.nn.leaky_relu(x)
        return x


class Discriminator(tf.keras.Model):

    def __init__(self):
        super(Discriminator, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)

        self.down1 = DiscDownsample(64, 4, False)
        self.down2 = DiscDownsample(128, 4)
        self.down3 = DiscDownsample(256, 4)

    # we are zero padding here with 1 because we need our shape to
    # go from (batch_size, 32, 32, 256) to (batch_size, 31, 31, 512)
        self.zero_pad1 = tf.keras.layers.ZeroPadding2D()
        self.conv = tf.keras.layers.Conv2D(512,
                                           (4, 4),
                                           strides=1,
                                           kernel_initializer=initializer,
                                           use_bias=False)
        self.batchnorm1 = tf.keras.layers.BatchNormalization()

    # shape change from (batch_size, 31, 31, 512) to (batch_size, 30, 30, 1)
        self.zero_pad2 = tf.keras.layers.ZeroPadding2D()
        self.last = tf.keras.layers.Conv2D(1,
                                           (4, 4),
                                           strides=1,
                                           kernel_initializer=initializer)

    # @tf.contrib.eager.defun
    def call(self, inp, tar, training):
        # concatenating the input and the target
        x = tf.concat([inp, tar], axis=-1)  # (bs, 256, 256, channels*2)
        x = self.down1(x, training=training)  # (bs, 128, 128, 64)
        x = self.down2(x, training=training)  # (bs, 64, 64, 128)
        x = self.down3(x, training=training)  # (bs, 32, 32, 256)

        x = self.zero_pad1(x)  # (bs, 34, 34, 256)
        x = self.conv(x)      # (bs, 31, 31, 512)
        x = self.batchnorm1(x, training=training)
        x = tf.nn.leaky_relu(x)

        x = self.zero_pad2(x)  # (bs, 33, 33, 512)
        # don't add a sigmoid activation here since
        # the loss function expects raw logits.
        x = self.last(x)      # (bs, cen, 30, 1 )

        return x


# The call function of Generator and Discriminator have been decorated
# with tf.contrib.eager.defun()
# We get a performance speedup if defun is used (~25 seconds per epoch)
generator = Generator()
discriminator = Discriminator()

# DEFINE THE LOSS FUNCTIONS AND THE OPTIMIZER

LAMBDA = 100


def discriminator_loss(disc_real_output, disc_generated_output):
    # real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(disc_real_output),
    #                                            logits=disc_real_output)
    real_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(disc_real_output),
                                                logits=disc_real_output)
    # real_loss = tf.losses.categorical_crossentropy(disc_real_output, disc_generated_output,
    #                                            from_logits=False, label_smoothing=0)

    # generated_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(disc_generated_output),
    #                                                 logits=disc_generated_output)
    generated_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(disc_generated_output),
                                                     logits=disc_generated_output)
    # generated_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(disc_generated_output),
    #                                                 logits=disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


def generator_loss(disc_generated_output, gen_output, target):
    # gan_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(disc_generated_output),
    #                                           logits=disc_generated_output)
    gan_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(disc_generated_output),
                                               logits=disc_generated_output)
    # gan_loss = tf.losses.categorical_crossentropy(gen_output, target,
    #                                            from_logits=False, label_smoothing=0)

  # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss


generator_optimizer = tf.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.optimizers.Adam(2e-4, beta_1=0.5)

# CHECKPOINTS (OBJECT-BASED SAVING)
checkpoint_dir = Dir_atual
checkpoint_prefix = os.path.join(
    checkpoint_dir, "checkpoints/")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# checkpoint_prefix

'''Training

    We start by iterating over the dataset
    The generator gets the input image and we get a generated output.
    The discriminator receives the input_image and the generated image as the first input. The second input is the input_image and the target_image.
    Next, we calculate the generator and the discriminator loss.
    Then, we calculate the gradients of loss with respect to both the generator and the discriminator variables(inputs) and apply those to the optimizer.

Generate Images

    After training, its time to generate some images!
    We pass images from the test dataset to the generator.
    The generator will then translate the input image into the output we expect.
    Last step is to plot the predictions and voila!'''

EPOCHS = 200


def generate_images(model, test_input, tar, save, address):

    global name
  # the training=True is intentional here since
  # we want the batch statistics while running the model
  # on the test dataset. If we use training=False, we will get
  # the accumulated statistics learned from the training dataset
  # (which we don't want)
    prediction = model(test_input, training=True)
  #  plt.figure(figsize=(15, 15))

  #  display_list = [test_input[0], tar[0], prediction[0]]
  #  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  #  for i in range(3):
  #      plt.subplot(1, 3, i+1)
  #      plt.title(title[i])
  #  # getting the pixel values between [0, 1] to plot it.
  #      plt.imshow(display_list[i] * 0.5 + 0.5)
  #      plt.axis('off')'''
    if save:
        plt.savefig(address+str(name)+'.jpg')
        name += 1
  #  plt.show()

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        print("RUNNING EPOCH: ", epoch+1, "of", epochs)
        cont_img = 1
        for input_image, target in dataset:
            if (cont_img % 10 == 0):
                print("Image: ", cont_img, "of", len(dataset))
            cont_img+=1
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                gen_output = generator(input_image, training=True)

                disc_real_output = discriminator(
                    input_image, target, training=True)
                disc_generated_output = discriminator(
                    input_image, gen_output, training=True)

                gen_loss = generator_loss(
                    disc_generated_output, gen_output, target)
                disc_loss = discriminator_loss(
                    disc_real_output, disc_generated_output)

            generator_gradients = gen_tape.gradient(gen_loss,
                                                    generator.variables)
            discriminator_gradients = disc_tape.gradient(disc_loss,
                                                        discriminator.variables)

            generator_optimizer.apply_gradients(zip(generator_gradients,
                                                    generator.variables))
            discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                        discriminator.variables))
            

        if epoch % 1 == 0:
            clear_output(wait=True)
            for inp, tar in valid_dataset.take(15):
                #generate_images(generator, inp, tar)
                address_save = Dir_atual + "/generated_images/"
                generate_images(generator, inp, tar, False , address_save)
                

        # saving (checkpoint) the model every 10 epochs
        #if (epoch + 1) % 20 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)
        print("Salving checkpoint in ", checkpoint_prefix)

        print("Dir Atual:", Dir_atual)
        print("Checkpoint Dir:", checkpoint_dir)
        print("Checkpoint Prefix:", checkpoint_prefix)
        print("Salvar em:", os.path.join(Dir_atual, "checkpoints/"))

        print('Time taken for epoch {} is {} sec\n'.format(
            epoch + 1, time.time()-start))

train(train_dataset, EPOCHS)

# restoring the latest checkpoint in checkpoint_dir 200_1500_3/  Croped_4284_200/
checkpoint.restore(tf.train.latest_checkpoint(
    os.path.join(Dir_atual, "checkpoints/")))

# Run the trained model on the entire test dataset
name = 0
address = '../datasets/covid19_2/test/A/'
for inp, tar in train_dataset.take(12):
    generate_images(generator, inp, tar, False, address)


# TESTING ON THE ENTIRE TEST DATASET
def save_generated_images(model1, test_input, tar1, address, gt_name):
    prediction1 = model1(test_input, training=True)
    A=prediction1[0] * 0.5 + 0.5
    # img=Image.fromarray(A)
    mpimg.imsave(address+str(gt_name), np.asarray(A))
    

# Run the trained model on the entire test dataset
#address = '../datasets/covid19_2/test/A/'
address = 'generated_images/'

# Run the trained model on the entire test dataset
c = 0
for inp, tar1, name in test_dataset.take(774):
    # GT_path=str(GT_path).split("'")[1]
    gt_name = str(name).split("'")[1].split("/")[-1]
    # print(gt_name)
    print("Test ", c, "/", len(test_dataset.take(774)))
    c+=1
    save_generated_images(generator, inp, tar1, address, gt_name)