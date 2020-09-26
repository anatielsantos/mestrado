from glob import glob
import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt

dataset_name = "covid-19"
data_type = "train"
path_image = glob('/home/anatielsantos/mestrado/bases/covid-19/images/*.nii')
path_mask_lung = glob('/home/anatielsantos/mestrado/bases/covid-19/lung_mask/*.nii')
path_mask_infec = glob('/home/anatielsantos/mestrado/bases/covid-19/infection_mask/*.nii')

def get_bounding_box_lung(image,intensity):
    image = sitk.Cast(image,sitk.sitkInt32)
    statistics = sitk.LabelStatisticsImageFilter()
    statistics.Execute(image,image)
    return statistics.GetBoundingBox(intensity)

def connected_components(sitk_image:sitk.Image):
    image_filter = sitk.ConnectedComponentImageFilter()
    return image_filter.Execute(sitk_image)

def get_roi_lung(path_mask, path_image):
    volume = len(path_mask)
    
    for v in range(volume):
        image_img_seg_roi = sitk.ReadImage(path_image[v])
        mask = sitk.ReadImage(path_mask[v])
        #mask_infec = sitk.ReadImage(path_mask_infec[v])
        
        #mask_nova = connected_components(mask) #extrai as lesões conectadas
        mask_nova = (mask>=1)*1

        print(str(len(np.unique(sitk.GetArrayFromImage(mask_nova))[1:])) + " regiões encontradas")
        for i in np.unique(sitk.GetArrayFromImage(mask_nova))[1:]:
            i = int(i)
            (min_x,max_x,min_y,max_y,min_z,max_z)=get_bounding_box_lung(mask_nova,i)
            print("VOL " + str(v) + " - ROI " + str(i) + " OK")
            
            # ROI Pulmão
            roi = image_img_seg_roi[min_x:max_x+1,min_y:max_y+1,min_z:max_z+1]
            
            # ROI Máscara
            #roi = mask_infec[min_x:max_x+1,min_y:max_y+1,min_z:max_z+1]
            
            sitk.WriteImage(roi,f"/home/anatielsantos/mestrado/bases/covid-19/roi_lung/image/image_vol{v}.nii")

get_roi_lung(path_mask_lung, path_mask_infec)