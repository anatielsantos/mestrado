import SimpleITK as sitk
import glob
import numpy as np

path_image = glob.glob("/home/anatielsantos/mestrado/bases/cortes-lidc/3d/image/solid/*.nii")
path_mask = glob.glob("/home/anatielsantos/mestrado/bases/cortes-lidc/3d/mask/solid/*.nii")

for i in range(len(path_image)):
    image = sitk.ReadImage(path_image[i])
    image_array = sitk.GetArrayFromImage(image)

    mask = sitk.ReadImage(path_mask[i])
    mask_array = sitk.GetArrayFromImage(mask)

    image_new = image_array
    pixel_min = image_array.min()

    for z in range(len(mask_array[:,0,0])):
        for y in range(len(mask_array[0,:,0])):
            for x in range(len(mask_array[0,0,:])):
                if not(mask_array[z,y,x] > 0):
                    image_new[z,y,x] = pixel_min
        
    img_sitk = sitk.GetImageFromArray(image_new)
    sitk.WriteImage(img_sitk, f"/home/anatielsantos/mestrado/bases/cortes-lidc/3d/image/voi/solid/lidc_voi_solid{i}.nii")

    print(f"ROI {i+1} de {len(path_image)} - OK")