import os

import cv2
import numpy as np
from skimage.util import random_noise

from src.nigep.utils.mkdir_folders import mkdir_dataset


def __write_images(dataset_name, noise_amount, image_path_arr):
    print(f'Generating with noise of {noise_amount} for {os.path.basename(image_path_arr)}')
    img = cv2.imread(image_path_arr)
    new_image_path = f'{os.getcwd()}/dataset/{dataset_name}/{os.path.basename(image_path_arr)}'
    if not os.path.exists(new_image_path):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        noise_img = random_noise(gray_image, mode='s&p', amount=noise_amount)
        noise_img = np.array(255 * noise_img, dtype='uint8')

        cv2.imwrite(new_image_path, noise_img)


def generate_dataset(x_data, dataset_name, noise_amount):
    mkdir_dataset(dataset_name)
    print('Copying images to the new dataset')
    [
        __write_images(dataset_name, noise_amount, path_array)
        for path_array in x_data
    ]
