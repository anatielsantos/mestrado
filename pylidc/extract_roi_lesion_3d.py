import SimpleITK as sitk
import glob
import numpy as np
#import nibabel as nib

# CT Lung and Infection Segmentation Dataset
path_mask1 = glob.glob("/home/anatielsantos/Documents/Mestrado/2020.1/Databases COVID-19/COVID-19 CT Lung and Infection Segmentation Dataset/Lung_and_Infection_Mask/*.nii")
path_mask_lung1 = glob.glob("/home/anatielsantos/Documents/Mestrado/2020.1/Databases COVID-19/COVID-19 CT Lung and Infection Segmentation Dataset/Lung_Mask/*.nii")

# CT segmentation dataset
path_mask2 = glob.glob("/home/anatielsantos/Documents/Mestrado/2020.1/Databases COVID-19/COVID-19 CT segmentation dataset/Segmentation_dataset_nr.2/rp_msk/*.nii")
path_mask_lung2 = glob.glob("/home/anatielsantos/Documents/Mestrado/2020.1/Databases COVID-19/COVID-19 CT segmentation dataset/Segmentation_dataset_nr.2/rp_lung_msk/*.nii")

# CT Lung Dataset
path_image1 = glob.glob("/home/anatielsantos/Documents/Mestrado/2020.1/Databases COVID-19/COVID-19 CT Lung and Infection Segmentation Dataset/COVID-19-CT-Seg_20cases/*.nii")

# CT segmentation dataset
path_image2 = glob.glob("/home/anatielsantos/Documents/Mestrado/2020.1/Databases COVID-19/COVID-19 CT segmentation dataset/Segmentation_dataset_nr.2/rp_im/*.nii")

# VOI infections
path_voi_infections1 = glob.glob("/home/anatielsantos/anaconda3/projetos/Fundamentos de Processamento Gráfico/covid19/roi_path_1/roi_infections/*.nii")
path_voi_infections2 = glob.glob("/home/anatielsantos/anaconda3/projetos/Fundamentos de Processamento Gráfico/covid19/roi_path_2/roi_infections/*.nii")

def get_bounding_box_lung(image,intensity):
    image = sitk.Cast(image,sitk.sitkInt32)
    statistics = sitk.LabelStatisticsImageFilter()
    statistics.Execute(image,image)
    return statistics.GetBoundingBox(intensity)

def connected_components(sitk_image:sitk.Image):
    image_filter = sitk.ConnectedComponentImageFilter()
    return image_filter.Execute(sitk_image)

def get_roi_exclusive(mask, image, vol, lesion):
    mask_array = sitk.GetArrayFromImage(mask)
    image_array = sitk.GetArrayFromImage(image)

    image_new = image_array
    pixel_min = image_new.min()

    for z in range(len(mask_array[:,0,0])):
        for y in range(len(mask_array[0,:,0])):
            for x in range(len(mask_array[0,0,:])):
                if not(mask_array[z,y,x] > 0):
                    image_new[z,y,x] = pixel_min
    
        img_sitk = sitk.GetImageFromArray(image_new)
        sitk.WriteImage(img_sitk, f"/home/anatielsantos/anaconda3/projetos/Fundamentos de Processamento Gráfico/covid19/roi_path_2/roi_infections_exclusive/vol{vol}_roi{lesion}.nii")

def get_roi_lung_infection(path_mask, path_image):
    volume = len(path_mask)
    for v in range(volume):
        mask = sitk.ReadImage(path_mask[v])
        #mask = (mask>2)*1 # comentar para rodar na base 2
        mask_nova = connected_components(mask)

        image_img_seg_roi = sitk.ReadImage(path_image[v])

        print(str(len(np.unique(sitk.GetArrayFromImage(mask_nova))[1:])) + " regiões encontradas")
        for i in np.unique(sitk.GetArrayFromImage(mask_nova))[1:]:
            i = int(i)
            (min_x,max_x,min_y,max_y,min_z,max_z)=get_bounding_box_lung(mask_nova,i)
            print("VOL " + str(v) + " - ROI " + str(i) + " OK")
            
            roi_ct = image_img_seg_roi[min_x:max_x+1,min_y:max_y+1,min_z:max_z+1]
            roi_mask = mask[min_x:max_x+1,min_y:max_y+1,min_z:max_z+1]
            get_roi_exclusive(roi_mask, roi_ct, v, i)

def get_roi_lung(path_mask, path_image):
    volume = len(path_mask)
    for v in range(volume):
        mask = sitk.ReadImage(path_mask[v])
        
        mask_nova = (mask>=1)*1
        image_img_seg_roi = sitk.ReadImage(path_image[v])

        print(str(len(np.unique(sitk.GetArrayFromImage(mask_nova))[1:])) + " regiões encontradas")
        for i in np.unique(sitk.GetArrayFromImage(mask_nova))[1:]:
            i = int(i)
            (min_x,max_x,min_y,max_y,min_z,max_z)=get_bounding_box_lung(mask_nova,i)
            print("VOL " + str(v) + " - ROI " + str(i) + " OK")
            roi = image_img_seg_roi[min_x:max_x+1,min_y:max_y+1,min_z:max_z+1]
            sitk.WriteImage(roi,f"/home/anatielsantos/anaconda3/projetos/Fundamentos de Processamento Gráfico/covid19/roi_path_2/roi_lung/lung_vol{v}.nii")

#get_roi_lung_infection(path_mask2, path_image2)
#get_roi_lung(path_mask1, path_image1)