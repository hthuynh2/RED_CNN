import os
import cv2
import scipy
import numpy as np
from scipy import ndimage

data_dir = './data'

def get_image_path(is_test, s, num):
    assert (s == 128 or s == 64)
    # path = os.path.join('/Users/hthieu/PycharmProjects/CS446_Final_Project', "xray_images/")
    path = os.path.join(os.getcwd(), "xray_images/")
    image_name = ""
    if is_test:
        path += 'test_images_'
        image_name += 'test_'
    else:
        path += 'train_images_'
        image_name += 'train_'
    if s == 64:
        path += '64x64'
    elif s == 128:
        path += '128x128'
    num_str = format(num, "05")
    image_name += num_str + ".png"
    return path+"/"+image_name

def imread(path):
    """
    Read image using its path.
    Default value is gray-scale, and image is read by YCbCr format as the paper said.
    """
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype('float')

def scale_image(img, factor=2.0):
    return ndimage.interpolation.zoom(img, factor, prefilter=False)

def load_data():
    input_file = os.path.join(data_dir, 'inputs.npy')
    label_file = os.path.join(data_dir, 'labels.npy')
    return np.load(input_file), np.load(label_file)
