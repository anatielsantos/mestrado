# GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import SimpleITK as sitk
import traceback
import glob
from tqdm import tqdm
from scipy.ndimage import morphology

from skimage.measure import label
from imageDivide import imageDivide
from skimage.filters import threshold_otsu

def fill_holes(binary_masks):
    binary_masks = morphology.binary_fill_holes(
    morphology.binary_dilation(
        morphology.binary_fill_holes(binary_masks > 0),
        iterations=1), structure=np.ones((3,13,13))
    ).astype(np.int)

    return binary_masks

def getLargestCC(segmentation):
    labels = label(segmentation)
    unique, counts = np.unique(labels, return_counts=True)
    list_seg=list(zip(unique, counts))[1:] # the 0 label is by default background so take the rest
    
    largest_1=max(list_seg, key=lambda x:x[1])[0]
    del(list_seg[largest_1 - 1])
    largest_2=max(list_seg, key=lambda x:x[1])[0]
    # del(list_seg[largest_2 - 1])
    # largest_3=max(list_seg, key=lambda x:x[1])[0]
    
    label_max_1=(labels == largest_1).astype(int)
    label_max_2=(labels == largest_2).astype(int)
    # label_max_3=(labels == largest_3).astype(int)

    return (label_max_2)

def get_voi_lung(exam_id, path_mask, output_path, bbox=False):
    try:
        print(exam_id + ':')

        image_mask = sitk.ReadImage(path_mask)
        image_array = sitk.GetArrayFromImage(image_mask).astype(np.int16)

        otsu = threshold_otsu(image_array)
        image_array = image_array < otsu
        
        reg1, reg2 = imageDivide(image_array)
        voi1 = getLargestCC(reg1)
        voi2 = getLargestCC(reg2)
        voi = np.concatenate([voi1, voi2], axis=2)

        voi = fill_holes(voi)

        if (bbox):
            new_image = np.zeros(voi.shape)
            for s in range(len(new_image[:,0,0])):
                menor_c =  640
                maior_c =  0
                menor_l =  640
                maior_l =  0
                print(s, "/", len(voi[:,0,0]))
                for l in range(len(voi[0,:,0])):
                    for c in range(len(voi[0,0,:])):
                        if ((voi[s,l,c] == 1) & (c < menor_c)):
                            menor_c = c
                        if ((voi[s,l,c] == 1) & (l < menor_l)):
                            menor_l = l
                        if ((voi[s,l,c] == 1) & (c > maior_c)):
                            maior_c = c
                        if ((voi[s,l,c] == 1) & (l > maior_l)):
                            maior_l = l
                        
                new_image[s, menor_l:maior_l+1,menor_c:maior_c+1] = image_array[s, menor_l:maior_l+1,menor_c:maior_c+1]
                voi = new_image

        itkImage = sitk.GetImageFromArray(voi.astype(np.int16))
        sitk.WriteImage(itkImage, output_path)

        del image_mask

    except Exception as e:
        print("type error: " + str(e))
        print(traceback.format_exc())
        return

def exec_get_voi_lung(mask_dir, dst_dir, ext, reverse = False, desc = None):
    try:
        os.stat(dst_dir)
    except:
        os.mkdir(dst_dir)

    input_mask_pathAll = glob.glob(mask_dir + '/*' + ext)
    input_mask_pathAll.sort(reverse=reverse)

    exam_ids = []
    input_src_paths = []
    input_mask_paths = []
    output_paths = []

    for input_path in input_mask_pathAll:
        exam_id = os.path.basename(input_path.replace(ext, ''))
        output_path = dst_dir + '/' + exam_id + '_voi' + ext

        # verifica se o arquivo ja existe
        if os.path.isfile(output_path):
            print('Arquivo ' + output_path + ' ja existe')
            # os.remove(output_path)
            continue

        exam_ids.append(exam_id)
        input_src_paths.append(input_path)
        output_paths.append(output_path)
    
    for input_mask_path in input_mask_pathAll:
        input_mask_paths.append(input_mask_path)

    for i, exam_id in enumerate(tqdm(exam_ids,desc=desc)):
        get_voi_lung(exam_id, input_mask_paths[i], output_paths[i], bbox=False)

def main():
    dataset = 'dataset2'
    ext = '.nii.gz' 
    main_mask_dir = f'/home/anatielsantos/mestrado/datasets/dissertacao/{dataset}/image/ZeroPedding/imagePositive'
    mask_dir = '{}'.format(main_mask_dir)
    dst_dir = '{}/VoiLung'.format(main_mask_dir)

    exec_get_voi_lung(mask_dir, dst_dir, ext, reverse = False, desc = f'Getting VOI from {dataset}')

if __name__=="__main__":
    main()