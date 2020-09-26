import SimpleITK as sitk
import glob
import numpy as np
from skimage.io import imsave
from skimage.filters import threshold_otsu
from matplotlib import pyplot as plt

path_image = glob.glob("/home/anatielsantos/Documents/Mestrado/2020.1/Databases COVID-19/COVID-19 CT Lung and Infection Segmentation Dataset/COVID-19-CT-Seg_20cases/*.nii")

# for i in range(len(path_image)):
image = sitk.ReadImage(path_image[0])
image_array = sitk.GetArrayFromImage(image[:,:,0])

otsu = threshold_otsu(image_array) / 255
image_otsu = image_array < otsu
print(image_otsu.shape)

imsave("/home/anatielsantos/Documents/Mestrado/2020.1/Databases COVID-19/COVID-19 CT Lung and Infection Segmentation Dataset/ct_otsu.nii", image_otsu)
# plt.imshow(image_otsu, cmap="gray")

# img_sitk = sitk.GetImageFromArray(image_otsu)
# sitk.WriteImage(img_sitk, "/home/anatielsantos/Documents/Mestrado/2020.1/Databases COVID-19/COVID-19 CT Lung and Infection Segmentation Dataset/ct_otsu.nii")