# GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import SimpleITK as sitk
import glob, time
from tqdm import tqdm
import traceback

def load_image(path_image, path_mask, remove_no_lesion = False):
    try:
        image = sitk.ReadImage(path_image)
        npyImage = sitk.GetArrayFromImage(image)

        mask = sitk.ReadImage(path_mask)
        npyMask = sitk.GetArrayFromImage(mask)

        # bin mask
        npyMask = (npyMask>0)*1
        npyMask = npyMask.astype(np.float32)

        # no lesion image remover
        if remove_no_lesion:
            remove_list_image = []
            remove_list_mask = []
            for i in range(npyImage.shape[0]):
                if np.amax(npyMask[i]) < 1:
                    remove_list_image.append(i)
                    remove_list_mask.append(i)
        
            print("Shape antes da remoção: ", npyImage.shape)
            # a = npyImage.shape[0]
            npyImage = np.delete(npyImage, remove_list_image, axis=0)
            npyMask = np.delete(npyMask, remove_list_mask, axis=0)
            # b = npyImage.shape[0]
            print(f"Shape depois da remoção de {len(remove_list_mask)} slices:", npyImage.shape)
            # print(f"Slices removidos {(a-b)}")

        # del(npyMask[i])
        # npyMask = np.delete(npyMask, i)

        del image
        del mask

        # itkImage = sitk.GetImageFromArray(npyImage)
        # sitk.WriteImage(itkImage, '/home/anatielsantos/mestrado/datasets/dissertacao/train/image/train_images.nii.gz')

        return npyImage, npyMask

    except Exception as e:
        print("type error: " + str(e))
        print(traceback.format_exc())
        return
    
def compress_dataset(src_dir, mask_dir, dst_dir, ext, joint, reverse = False, desc = None, remove_no_lesion = False):
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

    output_path = dst_dir
    
    for input_path in input_src_pathAll:
        exam_id = os.path.basename(input_path.replace(ext, ''))

        exam_ids.append(exam_id)
        input_src_paths.append(input_path)
        output_paths.append(output_path)

    for input_mask_path in input_mask_pathAll:
        input_mask_paths.append(input_mask_path)

    list_images, list_masks = list(), list()
    for i, exam_id in enumerate(tqdm(exam_ids,desc=desc)):
        images, masks = load_image(input_src_paths[i], input_mask_paths[i], remove_no_lesion=remove_no_lesion)
        list_images.append(images)
        list_masks.append(masks)

    np.savez_compressed(f"{output_path}/{joint}_images_exp5_lesion", list_images)
    np.savez_compressed(f"{output_path}/{joint}_masks_exp5_lesion", list_masks)

    # np.save(f"{output_path}/{joint}_images.npy", list_images)
    # np.save(f"{output_path}/{joint}_masks.npy", list_masks)
            
def main():
    ext = '.nii.gz'
    joint = 'train' # [train, test]
    main_dir_image = f'/home/anatielsantos/mestrado/datasets/dissertacao/final_tests_dis/teste5/Train'
    main_dir_mask = f'/home/anatielsantos/mestrado/datasets/dissertacao/final_tests_dis/teste5/Train_mask'
    
    src = main_dir_image
    tar = main_dir_mask
    src_dir = '{}'.format(src)
    mask_dir = '{}'.format(tar)
    
    dst_dir = '/home/anatielsantos/mestrado/datasets/dissertacao/final_tests_dis/teste5'

    compress_dataset(src_dir, mask_dir, dst_dir, ext, joint, reverse = False, desc = f'Compressing {joint} datasets', remove_no_lesion=True)

if __name__=="__main__":    
    start = time.time()
    main()
    stop = time.time()
    print("Elapsed time: "+str(stop - start)+" seconds")
