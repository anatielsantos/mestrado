# GPU
import os
import numpy as np
import SimpleITK as sitk
import glob
import time
from tqdm import tqdm
import traceback

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# width = new image width (needs to be bigger than current)
def zero_pad(width, exam_id, input_path, output_path):
    try:
        print(exam_id + ':')

        image = sitk.ReadImage(input_path)
        npyImage = sitk.GetArrayFromImage(image)
        npyImageCrop = np.zeros((npyImage.shape[0], 544, 544))

        print(npyImage.shape)
        print(npyImageCrop.shape)

        for i in range(npyImage.shape[0]):
            npyImageCrop[i] = npyImage[i][48:592, 48:592]

        npyImageCrop = npyImageCrop.astype(np.int16)

        itkImage = sitk.GetImageFromArray(npyImageCrop)
        sitk.WriteImage(itkImage, output_path)

        del image

    except Exception as e:
        print("type error: " + str(e))
        print(traceback.format_exc())
        return


def exec_zero_padding(src_dir, dst_dir, ext, width, reverse=False, desc=None):
    try:
        os.stat(dst_dir)
    except:
        os.mkdir(dst_dir)

    input_pathAll = glob.glob(src_dir + '/*' + ext)
    input_pathAll.sort(reverse=reverse)

    exam_ids = []
    input_paths = []
    output_paths = []

    for input_path in input_pathAll:
        exam_id = os.path.basename(input_path.replace(ext, ''))
        output_path = dst_dir + '/' + exam_id + '_crop' + ext

        # verifica se o arquivo ja existe
        if os.path.isfile(output_path):
            print('Arquivo ' + output_path + ' ja existe.')
            continue

        exam_ids.append(exam_id)
        input_paths.append(input_path)
        output_paths.append(output_path)

    for i, exam_id in enumerate(tqdm(exam_ids, desc=desc)):
        zero_pad(width, exam_id, input_paths[i], output_paths[i])


def main():
    ext = '.nii.gz'
    width = 640  # new width
    main_dir = '/home/anatielsantos/mestrado/datasets/dissertacao/dataset1/lesion_mask/ZeroPedding'

    src_dir = '{}'.format(main_dir)
    dst_dir = '{}/crop'.format(main_dir)

    exec_zero_padding(
        src_dir,
        dst_dir,
        ext,
        width,
        reverse=False,
        desc='croping'
    )


if __name__ == "__main__":
    start = time.time()
    main()
    stop = time.time()
    print("Elapsed time: "+str(stop - start)+" seconds")
