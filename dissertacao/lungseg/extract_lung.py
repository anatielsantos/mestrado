# GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import SimpleITK as sitk
import glob, time
from tqdm import tqdm
import traceback

def get_bounding_box_lung(image,intensity):
    image = sitk.Cast(image,sitk.sitkInt32)
    statistics = sitk.LabelStatisticsImageFilter()
    statistics.Execute(image,image)
    return statistics.GetBoundingBox(intensity)

# extract pulmonary parenchyma
def extract_lung(exam_id, src_path, mask_path, output_path, bbox=False):
    try:
        print(exam_id + ':')
        
        image = sitk.ReadImage(src_path)
        npyImage = sitk.GetArrayFromImage(image)
        mask = sitk.ReadImage(mask_path)
        npyMask = sitk.GetArrayFromImage(mask)

        # new_image = npyImage
        # for s in range(len(new_image[:,0,0])):
        #     if s % 10 == 0:
        #         print(s, "/", len(new_image[:,0,0]))
        #     for l in range(len(new_image[0,:,0])):
        #         for c in range(len(new_image[0,0,:])):
        #             if (npyMask[s,l,c] < 1):
        #                 new_image[s,l,c] = 0
        
        imgMin = np.amin(npyImage)
        npyImage_aux = npyImage
        if (imgMin < 0):
            imgMin = imgMin * -1
            npyImage_aux = npyImage + imgMin

        if bbox:
            # for i in range(npyMask.shape[0]):
            print(str(len(np.unique(sitk.GetArrayFromImage(mask))[1:])) + " regiões encontradas")
            for i in np.unique(sitk.GetArrayFromImage(mask))[1:]:
                i = int(i)
                (min_x,max_x,min_y,max_y,min_z,max_z)=get_bounding_box_lung(mask,i)
                npyImage_aux = npyImage_aux[min_x:max_x+1,min_y:max_y+1,min_z:max_z+1]
                itkImage = sitk.GetImageFromArray(np.asarray(npyImage_aux))
        else:
            itkImage = sitk.GetImageFromArray((npyImage_aux * npyMask))
            
        sitk.WriteImage(itkImage, output_path)

        del image

    except Exception as e:
        print("type error: " + str(e))
        print(traceback.format_exc())
        return
    
def exec_extract_lung(src_dir, mask_dir, dst_dir, ext, reverse = False, desc = None):
    try:
        os.stat(dst_dir)
    except:
        os.mkdir(dst_dir)    

    input_src_pathAll = glob.glob(src_dir + '/*' + ext)
    input_src_pathAll.sort(reverse=reverse) 

    input_mask_pathAll = glob.glob(mask_dir + '/*' + ext)
    input_mask_pathAll.sort(reverse=reverse) 

    exam_ids = []
    input_src_paths = []
    input_mask_paths = []
    output_paths = []

    for input_path in input_src_pathAll:
        exam_id = os.path.basename(input_path.replace(ext, ''))
        output_path = dst_dir + '/' + exam_id + '_LungExtract' + ext

        # verifica se o arquivo ja existe
        if os.path.isfile(output_path):
            print('Arquivo ' + output_path + ' ja existe e será removido')
            os.remove(output_path)
            # continue

        exam_ids.append(exam_id)
        input_src_paths.append(input_path)
        output_paths.append(output_path)
    
    for input_mask_path in input_mask_pathAll:
        input_mask_paths.append(input_mask_path)

    for i, exam_id in enumerate(tqdm(exam_ids,desc=desc)):
        extract_lung(exam_id, input_src_paths[i], input_mask_paths[i], output_paths[i], bbox=True)
            
def main():
    joint = 'test'
    dataset = 'dataset1'
    ext = '.nii.gz'
    # main_dir = f'/home/anatielsantos/mestrado/datasets/dissertacao/{joint}/image' 
    main_dir = f'/home/anatielsantos/mestrado/datasets/dissertacao/{dataset}/image/ZeroPedding'
    # main_mask_dir = f'/home/anatielsantos/mestrado/datasets/dissertacao/{joint}/mask/lung_mask'
    main_mask_dir = f'/home/anatielsantos/mestrado/datasets/dissertacao/{dataset}/lung_mask'
    
    src_dir = '{}'.format(main_dir)
    dst_dir = '{}/bbox'.format(main_dir)

    mask_dir = '{}'.format(main_mask_dir)

    exec_extract_lung(src_dir, mask_dir, dst_dir, ext, reverse = False, desc = f'Extracting lung from {joint}')

if __name__=="__main__":    
    start = time.time()
    main()
    stop = time.time()
    print("Elapsed time: "+str(stop - start)+" seconds")