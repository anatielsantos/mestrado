# GPU
import os
import numpy as np
import SimpleITK as sitk
import glob
import time
import traceback
from numpy.core.fromnumeric import amax
from sklearn.model_selection import KFold
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_image(path_image, path_mask, remove_no_lesion=False):
    try:
        image = sitk.ReadImage(path_image)
        npyImage = sitk.GetArrayFromImage(image)

        mask = sitk.ReadImage(path_mask)
        npyMask = sitk.GetArrayFromImage(mask)

        # bin mask
        npyMask = (npyMask > 0) * 1
        npyMask = npyMask.astype(np.float32)

        # no lesion image remover
        if remove_no_lesion:
            remove_list_image = []
            remove_list_mask = []
            for i in range(npyImage.shape[0]):
                if np.amax(npyMask[i]) < 1:
                    remove_list_image.append(i)
                    remove_list_mask.append(i)

            print("Shape antes da remoção: ", npyImage.shape, " | Exame:", path_image.split("/")[-1])
            # a = npyImage.shape[0]
            npyImage = np.delete(npyImage, remove_list_image, axis=0)
            npyMask = np.delete(npyMask, remove_list_mask, axis=0)
            # b = npyImage.shape[0]
            print(f"Shape depois da remoção de {len(remove_list_mask)} slices:", npyImage.shape)
            # print(f"Slices removidos {(a-b)}")

        del image
        del mask

        # itkImage = sitk.GetImageFromArray(npyImage)
        # sitk.WriteImage(itkImage, '/home/anatielsantos/mestrado/datasets/dissertacao/train/image/train_images.nii.gz')

        return npyImage, npyMask

    except Exception as e:
        print("type error: " + str(e))
        print(traceback.format_exc())
        return


def compress_dataset(
    src_dir,
    mask_dir,
    dst_dir,
    ext,
    reverse=False,
    desc=None,
    remove_no_lesion=False,
    kfold=False
):

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
    for i, exam_id in enumerate(tqdm(exam_ids, desc=desc)):
        images, masks = load_image(
            input_src_paths[i],
            input_mask_paths[i],
            remove_no_lesion=remove_no_lesion
        )

        list_images.append(images)
        list_masks.append(masks)

    # k-fold
    if kfold:
        kf = KFold(n_splits=10)
        kf.get_n_splits(input_src_pathAll)

        i = 0
        for train_index, test_index in kf.split(input_src_pathAll):
            print("TRAIN:", train_index, "TEST:", test_index)

            list_images_fold = np.delete(list_images, test_index[0], axis=0)
            list_masks_fold = np.delete(list_masks, test_index[0], axis=0)

            np.savez_compressed(f"{output_path}/images_mixed_quant_k{i}", list_images_fold)
            np.savez_compressed(f"{output_path}/masks_mixed_quant_k{i}", list_masks_fold)
            i = i + 1
    else:
        np.savez_compressed(f"{output_path}/images_mixed", list_images)
        np.savez_compressed(f"{output_path}/masks_mixed", list_masks)


def main():
    ext = '.nii.gz'
    main_dir_image = '/data/flavio/anatiel/datasets/dissertacao/final_tests/kfold/dataset_mixed/quant/images'
    main_dir_mask = '/data/flavio/anatiel/datasets/dissertacao/final_tests/kfold/dataset_mixed/quant/masks'

    # local
    # main_dir_image = '/home/anatielsantos/mestrado/datasets/dissertacao/bbox/quant/dataset*/images'
    # main_dir_mask = '/home/anatielsantos/mestrado/datasets/dissertacao/bbox/quant/dataset*/masks'

    src = main_dir_image
    tar = main_dir_mask
    src_dir = '{}'.format(src)
    mask_dir = '{}'.format(tar)

    dst_dir = '/data/flavio/anatiel/datasets/dissertacao/final_tests/kfold/dataset_mixed/quant'
    
    # local
    # dst_dir = '/home/anatielsantos/mestrado/datasets/dissertacao/bbox/quant/mixed_dataset'

    compress_dataset(
        src_dir,
        mask_dir,
        dst_dir,
        ext,
        reverse=False,
        desc=f'Compressing datasets',
        remove_no_lesion=True,
        kfold=True
    )


if __name__ == "__main__":
    start = time.time()
    main()
    stop = time.time()
    print("Elapsed time: "+str(stop - start)+" seconds")
