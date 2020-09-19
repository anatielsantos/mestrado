import numpy as np
import SimpleITK as sitk
from skimage.io import imsave
from skimage.color import rgb2gray
from skimage import img_as_float
import cv2

'''# BINARIZAÇÃO
image = sitk.ReadImage("img0_slc90.jpg")
image_array = sitk.GetArrayFromImage(image)

info = np.iinfo(image_array.dtype) # Get the information of the incoming image type
image_array = image_array.astype(np.float64) / info.max # normalize the image to 0 - 1
image_array = 255 * image_array # Now scale by 255
img = image_array.astype(np.uint8)
cv2.imshow("Binarizacao da imagem", img)
cv2.waitKey(0)

#new_name_img = "test_bw"
#imsave(f"{new_name_img}.jpg", img_gray, check_contrast=False)'''


# BINARIZAÇÃO COM OPENCV
img_ = cv2.imread('img0_slc90.jpg')
img_out = cv2.imread('/home/anatielsantos/workspace_visual/pix2pix-tensorflow/covid_test/images/img8_slc98-outputs.png')
img = img_
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
suave = cv2.GaussianBlur(img, (7, 7), 0) # aplica blur 
(T, bin) = cv2.threshold(suave, 160, 255, cv2.THRESH_BINARY)
(T, binI) = cv2.threshold(suave, 160, 255, cv2.THRESH_BINARY_INV)
#resultado = np.vstack([
#    np.hstack([suave, bin]),
#    np.hstack([binI, cv2.bitwise_and(img, img, mask = binI)])
#    ]) 

#new_name_img = "test_bw"
#imsave(f"{new_name_img}.jpg", bin, check_contrast=False)

cv2.imshow("Binarizacao da imagem", img_out)
cv2.waitKey(0)