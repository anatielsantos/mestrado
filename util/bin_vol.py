# GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import SimpleITK as sitk
from imageDivide import imageDivide
import traceback
import glob
from tqdm import tqdm
from skimage.filters import threshold_otsu
from scipy.ndimage.morphology import binary_closing


def get_voi_lung(exam_id, path_image, output_path):
    try:
        print(exam_id + ':')

        image = sitk.ReadImage(path_image)
        image_array = sitk.GetArrayFromImage(image)

        otsu = threshold_otsu(image_array)
        
        # print("OTSU", otsu)
        # print(np.amin(image_array))
        # print(np.amax(image_array))
        # # image_array = (image_array < otsu) * 0
        # for s in range(len(image_array[:,0,0])):
        #     for l in range(len(image_array[0,:,0])):
        #         for c in range(len(image_array[0,0,:])):
        #             if (image_array[s,l,c] >= otsu):
        #                 image_array[s,l,c] = 0

        # itkImage2 = sitk.GetImageFromArray(image_array.astype(float))
        # sitk.WriteImage(itkImage2, output_path)
        # print("imagem salva")
        
        image_otsu = image_array < otsu

        # image_otsu_dilate = binary_closing(image_otsu, structure=np.ones((1,13,13))).astype(image_otsu.dtype)
        # image_otsu_dilate = binary_closing(image_otsu).astype(image_otsu.dtype)
        image_otsu_dilate = image_otsu

        # BBox

        new_image = np.zeros(image_array.shape)
        # new_image = image_otsu_dilate
        for s in range(len(new_image[:,0,0])):
            menor_c =  640
            maior_c =  0
            menor_l =  640
            maior_l =  0
            print(s, "/", len(image_otsu_dilate[:,0,0]))
            for l in range(len(image_otsu_dilate[0,:,0])):
                for c in range(len(image_otsu_dilate[0,0,:])):
                    if ((image_otsu_dilate[s,l,c] == 0) & (c < menor_c)):
                        menor_c = c
                    if ((image_otsu_dilate[s,l,c] == 0) & (l < menor_l)):
                        menor_l = l
                    if ((image_otsu_dilate[s,l,c] == 0) & (c > maior_c)):
                        maior_c = c
                    if ((image_otsu_dilate[s,l,c] == 0) & (l > maior_l)):
                        maior_l = l
                    
            new_image[s, menor_l:maior_l+1,menor_c:maior_c+1] = image_otsu_dilate[s, menor_l:maior_l+1,menor_c:maior_c+1]
        
            itkImage = sitk.GetImageFromArray(new_image.astype(float))
            sitk.WriteImage(itkImage, output_path)

        del image

    except Exception as e:
        print("type error: " + str(e))
        print(traceback.format_exc())
        return

def exec_get_voi_lung(src_dir, dst_dir, ext, reverse = False, desc = None):
    try:
        os.stat(dst_dir)
    except:
        os.mkdir(dst_dir)    

    input_pathAll = glob.glob(src_dir + '/*' + ext)
    input_pathAll.sort(reverse=reverse) 

    exam_ids = []
    input_src_paths = []
    output_paths = []

    for input_path in input_pathAll:
        exam_id = os.path.basename(input_path.replace(ext, ''))
        output_path = dst_dir + '/' + exam_id + '_bin' + ext

        # verifica se o arquivo ja existe
        if os.path.isfile(output_path):
            print('Arquivo ' + output_path + ' ja existe')
            # os.remove(output_path)
            continue

        exam_ids.append(exam_id)
        input_src_paths.append(input_path)
        output_paths.append(output_path)

    for i, exam_id in enumerate(tqdm(exam_ids,desc=desc)):
        get_voi_lung(exam_id, input_src_paths[i], output_paths[i])

def main():
    ext = '.nii.gz' 
    main_dir = f'/home/anatielsantos/mestrado/datasets/dissertacao/dataset2/image/ZeroPedding/imagePositive'
    
    dst_dir = '{}/BinVolBB'.format(main_dir)

    exec_get_voi_lung(main_dir, dst_dir, ext, reverse = False, desc = f'Getting VOI')

if __name__=="__main__":
    main()