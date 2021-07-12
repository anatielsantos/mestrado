# GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import SimpleITK as sitk
import numpy as np
import cv2

def connected_components(sitk_image:sitk.Image):
    image_filter = sitk.ConnectedComponentImageFilter()
    return image_filter.Execute(sitk_image)

def connected_components_cv(path_image):
    #print(path_image.shape)
    image = cv2.imread(path_image, cv2.IMREAD_UNCHANGED)
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # getting mask with connectComponents
    ret, labels = cv2.connectedComponents(image)
    for label in range(1,ret):
        mask = np.array(labels, dtype=np.uint8)
        mask[labels == label] = 255
        #cv2.imshow('component',mask)
        #cv2.waitKey(0)

    print("Ret: ", ret)
    print("Labels: ", labels)
        
    # getting ROIs with findContours
    contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    for cnt in contours:
        (x,y,w,h) = cv2.boundingRect(cnt)
        ROI = image[y:y+h,x:x+w]
        #cv2.imshow('ROI', ROI)
        #cv2.waitKey(0)

    #cv2.destroyAllWindows()

if __name__ == "__main__":
    # settings
    pre_set = ''
    op = ''
    #path_preds = np.load('/data/flavio/anatiel/preds/unet/best/imgs_mask_test'+pre_set+op+'_none.npy').astype(np.float32)
    #path_mask_test = np.expand_dims(np.load('/data/flavio/anatiel/datasets/B_lesion/512x512/test.npz')['arr_0'].astype(np.float32), axis=-1)
    
    path_preds = '/data/flavio/anatiel/preds/unet/best/imgs_mask_test'+pre_set+op+'_none.npy'
    path_mask_test = '/data/flavio/anatiel/datasets/B_lesion/512x512/test.npz'
    
    #print(path_preds.shape)
    #print(path_mask_test.shape)
    
    #for i in range(path_preds.shape[0]):
    #pred = sitk.ReadImage(np.asarray(path_preds[i]))
    lesions_pred = connected_components_cv(path_preds)

    #mask = sitk.ReadImage(path_mask_test[i])
    #lesions_mask = connected_components(mask)

    #print("Slice:", i)
    print("Lesions Pred: ", len(lesions_pred))
    #print("Lesions Mask: ", len(lesions_mask))