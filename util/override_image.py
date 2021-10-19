# GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import SimpleITK as sitk
import glob, time
from tqdm import tqdm
import traceback

def imadjust(x,a,b,c,d,gamma=1):
    print("Nomalizing...")
    # Similar to imadjust in MATLAB.
    # Converts an image range from [a,b] to [c,d].
    # The Equation of a line can be used for this transformation:
    #   y=((d-c)/(b-a))*(x-a)+c
    # However, it is better to use a more generalized equation:
    #   y=((x-a)/(b-a))^gamma*(d-c)+c
    # If gamma is equal to 1, then the line equation is used.
    # When gamma is not equal to 1, then the transformation is not linear.

    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y

def override_image(exam_id, src_path, mask_path, output_path):
    try:
        print(exam_id + ':')
        
        image = sitk.ReadImage(src_path)
        npyImage = sitk.GetArrayFromImage(image)
        npyImage = imadjust(npyImage,np.amin(npyImage),np.amax(npyImage),0,1,gamma=1)

        mask = sitk.ReadImage(mask_path)
        npyMask = sitk.GetArrayFromImage(mask)

        print(np.amax(npyImage))
        print(np.amax(npyMask))

        # trocar valores > 0 da imagem com BBox e substituir pelos valores da imagem equalizada
        for s in range(len(npyImage[:,0,0])):
            for l in range(len(npyImage[0,:,0])):
                for c in range(len(npyImage[0,0,:])):
                    if (npyImage[s,l,c] > 0):
                        npyImage[s,l,c] = npyMask[s,l,c]

        itkImage = sitk.GetImageFromArray(npyImage)
        sitk.WriteImage(itkImage, output_path)

        del image

    except Exception as e:
        print("type error: " + str(e))
        print(traceback.format_exc())
        return
    
def exec_override_image(src_dir, mask_dir, dst_dir, ext, reverse = False, desc = None):
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
        output_path = dst_dir + '/' + exam_id + '_override' + ext

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
        override_image(exam_id, input_src_paths[i], input_mask_paths[i], output_paths[i])
            
def main():
    dataset = 'dataset2'
    ext = '.nii.gz'
    main_dir = f'/home/anatielsantos/mestrado/datasets/dissertacao/{dataset}/image/ZeroPedding/imagePositive/VoiLungBB'
    main_mask_dir = f'/home/anatielsantos/mestrado/datasets/dissertacao/{dataset}/image/ZeroPedding/imagePositive/VoiLungBB/equalizeHist'
    
    src_dir = '{}'.format(main_dir)
    dst_dir = '{}/overrided'.format(main_dir)

    mask_dir = '{}'.format(main_mask_dir)

    exec_override_image(src_dir, mask_dir, dst_dir, ext, reverse = False, desc = f'Overriding image')

if __name__=="__main__":    
    start = time.time()
    main()
    stop = time.time()
    print("Elapsed time: "+str(stop - start)+" seconds")