"""
num_channels.py
Find number of channels in your image.
Author: liuhh02 https://machinelearningtutorials.weebly.com/
"""

from PIL import Image
import numpy as np

# name of your image file
filename = '/home/anatielsantos/workspace_visual/datasets/covid19/B/train/img0_slc40.jpg'

# open image using PIL
img = Image.open(filename)

# convert to numpy array
img = np.array(img)

# find number of channels
if img.ndim == 2:
    channels = 1
    print("image has 1 channel")
else:
    channels = img.shape[-1]
    print("image has", channels, "channels")