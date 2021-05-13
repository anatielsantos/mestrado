import SimpleITK as sitk
import glob
import numpy as np
from skimage.transform import resize
from skimage.io import imsave
from skimage.exposure import equalize_adapthist
import cv2

from load_data import load_train_data, load_test_data, load_val_data

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# load data
# images
path_images_train = glob.glob("/home/anatielsantos/mestrado/bases/covid-19-nii/images/train/*.nii")
path_images_test = glob.glob("/home/anatielsantos/mestrado/bases/covid-19-nii/images/test/*.nii")
path_images_val = glob.glob("/home/anatielsantos/mestrado/bases/covid-19-nii/images/val/*.nii")
print("Images")
print("Train: ", len(path_images_train), "imagens")
print("Test: ", len(path_images_test), "imagens")
print("Val: ", len(path_images_val), "imagens")

# masks
path_masks_train = glob.glob("/home/anatielsantos/mestrado/bases/covid-19-nii/lung_mask/train/*.nii")
path_masks_test = glob.glob("/home/anatielsantos/mestrado/bases/covid-19-nii/lung_mask/test/*.nii")
path_masks_val = glob.glob("/home/anatielsantos/mestrado/bases/covid-19-nii/lung_mask/val/*.nii")
print("Masks")
print("Train: ", len(path_masks_train), "masks")
print("Test: ", len(path_masks_test), "masks")
print("Val: ", len(path_masks_val), "masks")

print("-"*30)

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

    return resized_image_array

# returns only images with lesion
# img_or_mask = save "img" or save "mask"
def masked(img_array, mask_array, img_or_mask):
    fatias = 0
    for slice_id in range(mask_array.shape[0]):
        if np.amax(mask_array[slice_id]) > 0:
             if np.amax(mask_array[slice_id]) == 3:
                fatias += 1

    masked_images = np.zeros((fatias,mask_array.shape[1],mask_array.shape[2]),dtype=np.float64)
    
    if img_or_mask == "img":
        save = img_array
    elif img_or_mask == "mask":
        save = mask_array

    s = 0
    for slice_id in range(img_array.shape[0]):
        if np.amax(mask_array[slice_id]) > 0:
            if np.amax(mask_array[slice_id]) == 3:
                masked_images[fatias - (fatias - s)]=save[slice_id]
                s += 1

    return masked_images

# blur
def blur_image(image_array):
    print("Blurring image...")

    image_array_blur = cv2.blur(image_array,(5,5))

    return image_array_blur

def imadjust(x,a,b,c,d,gamma=1):
    # Similar to imadjust in MATLAB.
    # Converts an image range from [a,b] to [c,d].
    # The Equation of a line can be used for this transformation:
    #   y=((d-c)/(b-a))*(x-a)+c
    # However, it is better to use a more generalized equation:
    #   y=((x-a)/(b-a))^gamma*(d-c)+c
    # If gamma is equal to 1, then the line equation is used.
    # When gamma is not equal to 1, then the transformation is not linear.

    print("Adjusting image [0 1]...")

    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y

def extract_lung(images, masks):
    print("Extracting lung...")
    new_image = images
    for s in range(len(new_image[:,0,0])):
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
        final_img[i] = equalize_adapthist(images[i])
                                 
    return final_img

# group = train, test, val
def preprocess(path_image, path_mask):
    images = None
    for i in range(len(path_image)):
        # read images
        img = sitk.ReadImage(path_image[i])
        img_array = sitk.GetArrayFromImage(img)

        # read masks
        mask = sitk.ReadImage(path_mask[i])
        mask_array = sitk.GetArrayFromImage(mask)

        # masked images
        # img_masked = masked(img_array, mask_array, "mask")

        # resize images
        # resized_image_array = resize_image(img_masked, 256, 256)

        # blur
        #img_blur = blur_image(img_array)

        # normalization 0 to 1
        img_float = imadjust(img_array,img_array.min(),img_array.max(),0,1)

        # clahe equalization
        clahe = equalize_clahe(img_float)

        # extract lung
        lung = extract_lung(clahe, mask_array)
        
        if images is None:
            images = lung
        else:
            images = np.concatenate([images,lung])

        # binarize image
        #images = (images > 0) * 1

    return images   

if __name__=="__main__":
    # extract lung

    # img_train, mask_train = load_train_data()
    # img_val, mask_val = load_val_data()
    # img_test, mask_test = load_test_data()
    
    # img_train_blur = blur_image(img_train)
    # img_val_blur = blur_image(img_val)
    # img_test_blur = blur_image(img_test)

    # images = extract_lung(img_train, mask_train)
    # images = extract_lung(img_train, mask_val)
    # images = extract_lung(img_test, mask_test)
    
    # images = preprocess(path_images_train, path_masks_train)
    # images = preprocess(path_images_val, path_masks_val)
    images = preprocess(path_images_test, path_masks_test)
    
    group = "test"
    subset = "A"
    print("Saving file...")
    np.savez_compressed(f"/home/anatielsantos/workspace_visual/mestrado/datasets/covid19/{subset}/512x512/lung_extracted/{group}_lung_clahe.npz",images)