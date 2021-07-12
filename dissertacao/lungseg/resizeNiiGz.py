import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"

from tqdm import tqdm

from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import SimpleITK as sitk
import glob
import time

def load_patient(image):
    imgs = np.load(image)
    imgs = imgs['arr_0']

    return imgs
    
def resize_image(exam_id, image, output_path, rows, cols):
    print(exam_id + ':')
    
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    npyImage = load_patient(image)

    print('-'*30)
    print('Resizing test data...')
    print('-'*30)

    resized_image_array = np.zeros((npyImage.shape[0],rows,cols),dtype=np.float64)
    #print(img_array.shape,np.unique(img_array))
    
    print(resized_image_array.shape)
    
    for slice_id in range(npyImage.shape[0]):
        resized_image_array[slice_id]=resize(npyImage[slice_id],(rows,cols),preserve_range=True)

    return resized_image_array
    
def execResize(src_dir, dst_dir, ext, reverse = False, desc = None, parallel = True):
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
        output_path = dst_dir + '/' + exam_id + '_resized' + ext   

        # verifica se o arquivo ja existe
        if os.path.isfile(output_path):
            print('Arquivo ' + output_path + ' ja existe.')
            continue

        exam_ids.append(exam_id)  
        input_paths.append(input_path)
        output_paths.append(output_path)   

    for i, exam_id in enumerate(tqdm(exam_ids,desc=desc)):
        resize_image(exam_id, input_paths[i], output_paths[i], 512, 512)

def main():

    ext = '.nii.gz'

    main_dir = '/data/flavio/anatiel/datasets/dissertacao'

    # src_dir = '{}/Volume'.format(main_dir)
    src_dir = '{}'.format(main_dir)
    dst_dir = '{}/Resized'.format(main_dir)       

    print("execResize")
    execResize(src_dir, dst_dir, ext, reverse = False, desc = 'Resizing exams', parallel=False)  

if __name__ == '__main__':
    arquivo = open("time_execution.txt", "a")
    start = time.time()
    main()
    stop = time.time()
    text = 'Execution time: ' + str(stop - start)
    arquivo.write(text)