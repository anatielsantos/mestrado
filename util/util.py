import SimpleITK as sitk
import glob
import numpy as np
from skimage.color import rgb2gray
from skimage.io import imsave

conjunto = "infection_mask"
path_train = glob.glob(f"/home/anatielsantos/mestrado/bases/covid-19/{conjunto}/train/*.nii")
path_test = glob.glob(f"/home/anatielsantos/mestrado/bases/covid-19/{conjunto}/test/*.nii")
path_val = glob.glob(f"/home/anatielsantos/mestrado/bases/covid-19/{conjunto}/val/*.nii")
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

# separa um volume em slices 2D
# path = conjunto de imagens a serem fatiadas
# group =  grupo (train, test, val)
# contagem inicial da imagem dentro do grupo (nomes sequenciais nos 3 grupos)
def save_slice(path_image, group, num_image = 0):
            
    for i in range(len(path_image)):

        img = sitk.ReadImage(path_image[i])
        img_array = sitk.GetArrayFromImage(img)
        new_slice = np.ones((img_array.shape[1], img_array.shape[2]))

        for p in range(img_array.shape[0]):
            for l in range(img_array.shape[1]):
                for c in range(img_array.shape[2]):
                    new_slice[l, c] = img_array[p, l, c]

            print("Salvando imagem ", i, "slice ", p)
            new_slice_gray = rgb2gray(new_slice)

            imsave(f"/home/anatielsantos/workspace_visual/datasets/covid-19/B/{group}/img{i+num_image}_slc{p}.jpg", new_slice_gray, check_contrast=False)

save_slice(path_train, "train", 0)
save_slice(path_test, "test", 6)
save_slice(path_val, "val", 8)