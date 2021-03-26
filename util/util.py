import SimpleITK as sitk
import glob
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.io import imsave
import cv2

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# images
path_images_train = glob.glob("/home/anatielsantos/mestrado/bases/covid-19-nii/images/train/*.nii")
path_images_test = glob.glob("/home/anatielsantos/mestrado/bases/covid-19-nii/images/test/*.nii")
path_images_val = glob.glob("/home/anatielsantos/mestrado/bases/covid-19-nii/images/val/*.nii")
print("Images")
print("Train: ", len(path_images_train), "imagens")
print("Test: ", len(path_images_test), "imagens")
print("Val: ", len(path_images_val), "imagens")

# masks
path_masks_train = glob.glob("/home/anatielsantos/mestrado/bases/covid-19-nii/infection_mask/train/*.nii")
path_masks_test = glob.glob("/home/anatielsantos/mestrado/bases/covid-19-nii/infection_mask/test/*.nii")
path_masks_val = glob.glob("/home/anatielsantos/mestrado/bases/covid-19-nii/infection_mask/val/*.nii")
print("Masks")
print("Train: ", len(path_masks_train), "masks")
print("Test: ", len(path_masks_test), "masks")
print("Val: ", len(path_masks_val), "masks")

# returns the largest shape among dataset images
def shape_larger(path_image):
    maior_x = 0
    maior_y = 0

    for i in range(path_image):
        image = sitk.ReadImage(path_image[i])
        image_array = sitk.GetArrayFromImage(image)
        if (image_array.shape[0] > maior_x):
            maior_x = image_array.shape[0]
        if (image_array.shape[1] > maior_y):
            maior_y = image_array.shape[1]
    
    return maior_x, maior_y

# save all the dataset images with the shape of the largest image
def reshape_center(path_image):
    x, y = shape_larger(path_image)

    for i in range(len(path_image)):
        new_image = np.zeros([x, y])

        name_img = path_image[i].split('/') 
        new_name_img = name_img[-1][0:-4]

        image = sitk.ReadImage(path_image[i])
        image_array = sitk.GetArrayFromImage(image)
        pad_x = (new_image.shape[0] - image_array.shape[0]) // 2
        pad_y = (new_image.shape[1] - image_array.shape[1]) // 2
        for l in range(image_array.shape[0]):
            for c in range(image_array.shape[1]):
                new_image[l + pad_x][c + pad_y] = image_array[l][c]
        imsave(f"/home/anatielsantos/workspace_visual/datasets/covid-19/B/val/{new_name_img}.jpg", new_image, check_contrast=False)
        print(f"Imagem {i} salva")

# resize images do rown, cols
def resize_image(img_array, rows, cols):
    resized_image_array = np.zeros((img_array.shape[0],rows,cols),dtype=np.float64)
    #print(img_array.shape,np.unique(img_array))
    
    for slice_id in range(img_array.shape[0]):
        resized_image_array[slice_id]=resize(img_array[slice_id],(rows,cols),preserve_range=True)
    print(resized_image_array.shape)

    return resized_image_array

# returns only images with lesion
def masked(img_array, mask_array):
    fatias = 0
    for slice_id in range(mask_array.shape[0]):
        if np.amax(mask_array[slice_id]) > 0:
            fatias += 1

    masked_images = np.zeros((fatias,mask_array.shape[1],mask_array.shape[2]),dtype=np.float64)

    s = 0
    for slice_id in range(img_array.shape[0]):
        if np.amax(mask_array[slice_id]) > 0:
            masked_images[fatias - (fatias - s)]=img_array[slice_id]
            s += 1
    print(masked_images.shape)

    return masked_images

# blur
def blur_image(image_array):
    image_array_blur = cv2.blur(image_array,(5,5))

    return image_array_blur

# group = train, test, val
def save_npz(path_image, path_mask, group):
    images = None
    for i in range(len(path_image)):
        # read images
        img = sitk.ReadImage(path_image[i])
        img_array = sitk.GetArrayFromImage(img)

        # read masks
        mask = sitk.ReadImage(path_mask[i])
        mask_array = sitk.GetArrayFromImage(mask)

        # masked images
        print("Masked images")
        img_masked = masked(img_array, mask_array)

        # resize images
        # print("Resizing start")
        # resized_image_array = resize_image(img_masked, 256, 256)

        # blur
        # print("Blurring start")
        # img_blur = blur_image(resized_image_array)
        
        if images is None:
            images=img_masked
        else:
            images = np.concatenate([images,img_masked])

        # binarize image
        #images = (images > 0) * 1
    print(images.shape)
    #np.savez_compressed(f"/home/anatielsantos/workspace_visual/mestrado/datasets/covid19/A/512x512/{group}_masked.npz",images)

save_npz(path_images_train, path_masks_train, "train")
save_npz(path_images_test, path_masks_test, "test")
save_npz(path_images_val, path_masks_val, "val")