from glob import glob
import numpy as np
import SimpleITK as sitk

# CÓDIGO ADICIONADO POR ANATIEL SANTOS PARA PRÉ-PROCESSAMENTO DAS IMAGENS
def maior_shape():
    maior_z = 272
    maior_x = 304
    maior_y = 432

    #path_image = glob('bases/covid-19/roi_lung/*/*/*.nii')
    #for i in range(len(path_image)):
    #    _img = sitk.ReadImage(path_image[i])
    #    _img_array = sitk.GetArrayFromImage(_img)
        
    #    for i in range(len(_img_array)):
            #if (_img_array.shape[0] > maior_z):
            #    maior_z = _img_array.shape[0]
    #        if (_img_array.shape[1] > maior_x):
    #            maior_x = _img_array.shape[1]
    #        if (_img_array.shape[2] > maior_y):
    #            maior_y = _img_array.shape[2]
    
    return maior_z, maior_x, maior_y

def normaliza_img(maior_z, maior_x, maior_y, image):   
    new_shape = np.ones((maior_z, maior_x, maior_y, 1))
    center_z = (maior_z - image.shape[0]) // 2
    center_x = (maior_x - image.shape[1]) // 2
    center_y = (maior_y - image.shape[2]) // 2
    for p in range(image.shape[0]):
        for l in range(image.shape[1]):
            for c in range(image.shape[2]):
                new_shape[p + center_z, l + center_x, c + center_y] = image[p,l,c]
    
    return new_shape

# separando slices 2D
# path = conjunto de imagens a serem fatiadas
# group =  grupo (train, test, val)
def save_slice(path, group):
    path_image = glob('bases/covid-19/roi_lung/%s/%s/*.nii' % (path, group))
        
    for i in range(len(path_image)):
        img = sitk.ReadImage(path_image[i])
        img_array = sitk.GetArrayFromImage(img)
        new_slice = np.ones((img_array.shape[1], img_array.shape[2]))

        for p in range(img_array.shape[0]):
            for l in range(img_array.shape[1]):
                for c in range(img_array.shape[2]):
                    new_slice[l, c] = img_array[p, l, c]

            print("Salvando imagem ", i, "slice ", p)
            new_slice_img = sitk.GetImageFromArray(new_slice)
            sitk.WriteImage(new_slice_img,f"/home/anatielsantos/workspace_visual/bases/covid-19/roi_lung/slices/{path}/{group}/img{i}_slc{p}.nii")


#dataset de treino
def load_data_train(dataset_name = "covid-19"):
    data_type = "train"
    path_image = glob('bases/%s/roi_lung/image/%s/*.nii' % (dataset_name, data_type))
    path_mask = glob('bases/%s/roi_lung/mask/%s/*.nii' % (dataset_name, data_type))
                        
    maior_z, maior_x, maior_y = maior_shape()
    imgs_train = []
    for i in range(len(path_image)):
        img = sitk.ReadImage(path_image[i])
        img_array = sitk.GetArrayFromImage(img)
        img_normalizada = normaliza_img(maior_z, maior_x, maior_y, img_array)
        imgs_train.append(img_normalizada)

    masks_train = []
    for i in range(len(path_mask)):
        mask = sitk.ReadImage(path_mask[i])
        mask_array = sitk.GetArrayFromImage(mask)
        mask_normalizada = normaliza_img(maior_z, maior_x, maior_y, mask_array)
        masks_train.append(mask_normalizada)
    
    return imgs_train, masks_train

#dataset de validação
def load_data_val(dataset_name = "covid-19"):
    data_type = "val"
    path_image = glob('bases/%s/roi_lung/image/%s/*.nii' % (dataset_name, data_type))
    path_mask = glob('bases/%s/roi_lung/mask/%s/*.nii' % (dataset_name, data_type))

    maior_z, maior_x, maior_y = maior_shape()
    imgs_val = []
    for i in range(len(path_image)):
        img = sitk.ReadImage(path_image[i])
        img_array = sitk.GetArrayFromImage(img)
        img_normalizada = normaliza_img(maior_z, maior_x, maior_y, img_array)
        imgs_val.append(img_normalizada)
    
    masks_val = []
    for i in range(len(path_mask)):
        mask = sitk.ReadImage(path_mask[i])
        mask_array = sitk.GetArrayFromImage(mask)
        mask_normalizada = normaliza_img(maior_z, maior_x, maior_y, mask_array)
        masks_val.append(mask_normalizada)

    return imgs_val, masks_val

save_slice("mask_infection", "val")
#load_data_train()
#load_data_val()