# GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import SimpleITK as sitk

def imageDivide(image):
    mid_image = np.int16(image.shape[1] / 2)

    reg1 = image[:, 0:image.shape[1], 0:mid_image]
    reg2 = image[:, 0:image.shape[1], mid_image:image.shape[2]]

    return reg1, reg2