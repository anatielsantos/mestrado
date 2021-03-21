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

conjunto = "lung_and_infection_mask"
path_train = glob.glob("/home/anatielsantos/mestrado/bases/covid-19-nii/infection_mask/train/*.nii")
path_test = glob.glob("/home/anatielsantos/mestrado/bases/covid-19-nii/infection_mask/test/*.nii")
path_val = glob.glob("/home/anatielsantos/mestrado/bases/covid-19-nii/infection_mask/val/*.nii")
print("Train: ", len(path_train), "imagens")
print("Test: ", len(path_test), "imagens")
print("Val: ", len(path_val), "imagens")

# retorna o maior shape entre as imagens do dataset
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

# salva todas as imagens do dataset com o shape da maior imagem
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

# resize 50%
# images = diretório das imagens originais
# path_sava = diretório para salvar as imagens redimensionadas
def resize_image(images, path_save, conjunto):
    print("Resizing start")
    for image in images:
        img = cv2.imread(image)

        name_img = image.split('/')
        new_name_img = name_img[-1][0:]

        scale_percent = 50
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img_resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        cv2.imwrite(path_save + conjunto + "/" + new_name_img, img_resized)
    print("Resizing end")

# separa um volume em slices 2D
# path = conjunto de imagens a serem fatiadas
# group =  grupo (train, test, val)
# contagem inicial da imagem dentro do grupo (nomes sequenciais nos 3 grupos)
def save_slice(path_image, group, num_image = 0):
    images = None
    for i in range(len(path_image)):
        img = sitk.ReadImage(path_image[i])
        img_array = sitk.GetArrayFromImage(img)
        
        # resize
        #resized_image_array = np.zeros((img_array.shape[0],512,512),dtype=np.float64)
        print(img_array.shape,np.unique(img_array))
        
        # for slice_id in range(img_array.shape[0]):
        #     resized_image_array[slice_id]=resize(img_array[slice_id],(256,256),preserve_range=True)
        # print(resized_image_array.shape,np.unique(resized_image_array))

        
        # if images is None:
        #     images=resized_image_array
        # else:
        #     images = np.concatenate([images,resized_image_array])

        if images is None:
            images=img_array
        else:
            images = np.concatenate([images, img_array])

        # binarizar imagem
        images = (images >0)*1
    print(images.shape)
    np.savez_compressed(f"/home/anatielsantos/workspace_visual/mestrado/datasets/covid19/B-lesion/512x512/{group}.npz",images)

# save_slice(path_train, "train", 0)
save_slice(path_test, "test", 6)
save_slice(path_val, "val", 8)