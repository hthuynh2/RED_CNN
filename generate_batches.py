import numpy as np
from utils import imread, get_image_path, scale_image
import os

PATCH_SIZE = 55
stride = 12
data_dir = './data'

def generate_batches():
    start_idx = 4000
    end_idx = 18000
    num_patch_per_image = 0

    for x in range(0, 128 - PATCH_SIZE, stride):
        for y in range(0, 128 - PATCH_SIZE, stride):
            num_patch_per_image += 1
    total_num_patch = num_patch_per_image * (end_idx - start_idx)
    data = np.zeros((total_num_patch, PATCH_SIZE, PATCH_SIZE, 1))
    labels = np.zeros((total_num_patch, PATCH_SIZE, PATCH_SIZE, 1))
    print("Total number of patches " + str(total_num_patch))
    cur_idx = 0
    for i in range(start_idx, end_idx):
        if start_idx % 500:
            print("Processing image number " + str(start_idx) + "...")
        noisy_img = imread(get_image_path(True, 64, i)) # Image size 64x64
        noisy_img = scale_image(noisy_img, 2.0) # Image size 128x128
        noisy_img /= 255.0
        clean_img = imread(get_image_path(True, 128, i)) # Image size 128x128
        clean_img /= 255.0
        im_h, im_w = noisy_img.shape

        for x in range(0, im_h - PATCH_SIZE, stride):
            for y in range(0, im_w - PATCH_SIZE, stride):
                clean_patch = clean_img[x:x + PATCH_SIZE, y:y + PATCH_SIZE]
                clean_patch = clean_patch.reshape([PATCH_SIZE, PATCH_SIZE, 1])
                labels[cur_idx] = clean_patch

                noisy_patch = noisy_img[x:x + PATCH_SIZE, y:y + PATCH_SIZE]
                noisy_patch = noisy_patch.reshape([PATCH_SIZE, PATCH_SIZE, 1])
                data[cur_idx] = noisy_patch

                cur_idx += 1

    assert cur_idx == total_num_patch - 1
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    print("Saving to files...")
    np.save(os.path.join(data_dir, "inputs.npy"), data)
    np.save(os.path.join(data_dir, "labels.npy"), labels)


if __name__ == '__main__':
    generate_batches()
