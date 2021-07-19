# GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import SimpleITK as sitk
import glob, time
from tqdm import tqdm
import traceback
from scipy.ndimage import morphology

# morphological operations ['median', 'erode', 'dilate', 'opening', 'closing']
def morphological_operations(im_mask_gan, op):
    kernel = np.ones((6,6),np.uint8)

    if op == 'none':
        operation = np.float32(im_mask_gan)
    if op == 'median':
        operation = cv2.medianBlur(np.float32(im_mask_gan), 5)
    if op == 'erode':
        operation = cv2.erode(np.float32(im_mask_gan), kernel, iterations=1)
    if op == 'dilate':
        operation = cv2.dilate(np.float32(im_mask_gan), kernel, iterations=1)
    if op == 'opening':
        operation = cv2.morphologyEx(np.float32(im_mask_gan), cv2.MORPH_OPEN, kernel)
    if op == 'closing':
        operation = morphology.binary_fill_holes(
            morphology.binary_closing(
                morphology.binary_fill_holes(im_mask_gan.astype(np.uint8) > 0).astype(int),
                iterations=1)
            )
    
    return operation

# extract pulmonary parenchyma
def morph(exam_id, mask_path, output_path):
    try:
        print(exam_id + ':')
        
        mask = sitk.ReadImage(mask_path)
        npyMask = sitk.GetArrayFromImage(mask)
        
        npyMask_closing = morphological_operations(npyMask, 'closing')

        itkImage = sitk.GetImageFromArray(npyMask_closing.astype(np.uint16))
        sitk.WriteImage(itkImage, output_path)

        del mask

    except Exception as e:
        print("type error: " + str(e))
        print(traceback.format_exc())
        return
    
def exec_morph(mask_dir, dst_dir, ext, reverse = False, desc = None):
    try:
        os.stat(dst_dir)
    except:
        os.mkdir(dst_dir)    

    input_mask_pathAll = glob.glob(mask_dir + '/*' + ext)
    input_mask_pathAll.sort(reverse=reverse) 

    exam_ids = []
    input_mask_paths = []
    output_paths = []

    for input_path in input_mask_pathAll:
        exam_id = os.path.basename(input_path.replace(ext, ''))
        output_path = dst_dir + '/' + exam_id + '_closing' + ext

        # verifica se o arquivo ja existe
        if os.path.isfile(output_path):
            print('Arquivo ' + output_path + ' ja existe.')
            continue

        exam_ids.append(exam_id)
        input_mask_paths.append(input_path)
        output_paths.append(output_path)

    for i, exam_id in enumerate(tqdm(exam_ids,desc=desc)):
        morph(exam_id, input_mask_paths[i], output_paths[i])
            
def main():
    ext = '.nii.gz'
    main_dir = '/home/anatielsantos/mestrado/datasets/dissertacao/dataset1/PulmoesZeroPedding' 
    main_mask_dir = '/home/anatielsantos/mestrado/datasets/dissertacao/dataset1/PulmoesZeroPedding/PulmoesMascaraUNet'
    
    src_dir = '{}'.format(main_dir)
    dst_dir = '{}/PulmoesMascara'.format(main_dir)

    mask_dir = '{}'.format(main_mask_dir)

    exec_morph(mask_dir, dst_dir, ext, reverse = False, desc = 'Closing Morph')

if __name__=="__main__":    
    # arquivo = open("anatiel/dissertacao/lungseg/time_execution_zero_padding.txt", "a")
    # start = time.time()
    main()
    # stop = time.time()
    # exec_time = stop - start
    # arquivo.write(str(exec_time))
