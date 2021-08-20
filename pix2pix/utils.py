import tensorflow as tf
import numpy as np
import pandas as pd
from skimage.exposure import equalize_adapthist
import cv2

# load dataset
def load_images(path_src, path_mask):
    # src = np.expand_dims(np.load(path_src)['arr_0'], axis=-1)
    # tar = np.expand_dims(np.load(path_mask)['arr_0'].astype(np.float32), axis=-1)
    
    # return [src,tar]

    src_npz = np.load(path_src, allow_pickle=True)
    tar_npz = np.load(path_mask, allow_pickle=True)
    src = src_npz['arr_0']
    tar = tar_npz['arr_0']
    
    # return np.float32(np.expand_dims(np.concatenate(src), axis=-1)), np.float32(np.expand_dims(np.concatenate(tar), axis=-1))
    return src, tar

def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image

def random_crop(input_image, real_image,IMG_HEIGHT, IMG_WIDTH):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
            stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]


# normalizing the images to [-1, 1]
def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image

@tf.function()
def random_jitter(input_image, real_image,IMG_HEIGHT=256, IMG_WIDTH=256):
    # resizing to 286 x 286 x 3
    input_image, real_image = resize(input_image, real_image, 286, 286)

    # randomly cropping to 256 x 256 x 3
    input_image, real_image = random_crop(input_image, real_image,IMG_HEIGHT, IMG_WIDTH)

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image

# normalize the images to [0, 1]
def imadjust(x,a,b,c,d,gamma=1):
    print("Nomalizing...")
    # Similar to imadjust in MATLAB.
    # Converts an image range from [a,b] to [c,d].
    # The Equation of a line can be used for this transformation:
    #   y=((d-c)/(b-a))*(x-a)+c
    # However, it is better to use a more generalized equation:
    #   y=((x-a)/(b-a))^gamma*(d-c)+c
    # If gamma is equal to 1, then the line equation is used.
    # When gamma is not equal to 1, then the transformation is not linear.

    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y

# extract pulmonary parenchyma
def extract_lung(images, masks):
    print("Lung extranting...")
    new_image = images
    for s in range(len(new_image[:,0,0])):
        if s % 10 == 0:
            print(s, "/", len(new_image[:,0,0]))
        for l in range(len(new_image[0,:,0])):
            for c in range(len(new_image[0,0,:])):
                if (masks[s,l,c] == 0):
                    new_image[s,l,c] = 0
                    
        #imsave(f"/home/anatielsantos/Desktop/tes{s}.jpg", new_image[s], check_contrast=False)
    return new_image

# clahe equalization
def equalize_clahe(images):
    print("Clahe equalization...")
    final_img = images
    for i in range(images.shape[0]):
        if i % 10 == 0:
            print(i, "/", images.shape[0])
        final_img[i][:,:,0] = equalize_adapthist(images[i][:,:,0])
                                 
    return final_img

# blur
def blur_image(images):
    print("Blurring images...")
    final_img = images
    for i in range(images.shape[0]):
        final_img[i][:,:,0] = cv2.blur(images[i][:,:,0],(5,5))
                                 
    return final_img
    
# results train
def results_train(history):
    amax = np.amax(history)
    mean = np.mean(history)
    
    return amax, mean

def bg_blck(images):
    print("BG Black...")
    new_image = images
    for s in range(len(new_image[:,0,0])):
        print(s, "/", len(new_image[:,0,0]))
        for l in range(len(new_image[0,:,0])):
            for c in range(len(new_image[0,0,:])):
                if (new_image[s,l,c] == np.amin(images[s])):
                    new_image[s,l,c] = 0
                    
    return new_image

# morphological operations ['median', 'erode', 'dilate', 'opening', 'closing']
def morphological_operations(im_mask_gan, op): 
    kernel = np.ones((6,6),np.uint8)

    if op == 'none':
        operation = np.float32(im_mask_gan)
    if op == 'median':
        operation = cv2.medianBlur(np.float32(im_mask_gan), 5)
    if op == 'erode':
        operation = cv2.erode(np.float32(im_mask_gan), kernel, iterations=1)
    if op == 'dilate':
        operation = cv2.dilate(np.float32(im_mask_gan), kernel, iterations=1)
    if op == 'opening':
        operation = cv2.morphologyEx(np.float32(im_mask_gan), cv2.MORPH_OPEN, kernel)
    if op == 'closing':
        operation = cv2.morphologyEx(np.float32(im_mask_gan), cv2.MORPH_CLOSE, kernel)
    
    return operation