import os
import time
import concurrent.futures

import cv2
import numpy as np
from skimage.util import random_noise

from ..utils.mkdir_folders import mkdir_dataset


def __write_image(dataset_name, noise_amount, image_path_arr):
    img = cv2.imread(image_path_arr)
    new_image_path = f'{os.getcwd()}/dataset/{dataset_name}/{os.path.basename(image_path_arr)}'
    if not os.path.exists(new_image_path):
        # gray_image = cv2.cvtColor(, cv2.COLOR_BGR2GRAY)
        noise_img = random_noise(img, mode='s&p', amount=noise_amount)
        noise_img = np.array(255 * noise_img, dtype='uint8')

        cv2.imwrite(new_image_path, noise_img)


def __write_noise_dataset(x_data, dataset_name, noise_amount):
    [__write_image(dataset_name, noise_amount, path_array) for path_array in x_data]


def generate_dataset(x_data, dataset_name, noise_amount):
    print(f'Generating dataset with noise of {noise_amount}')
    mkdir_dataset(dataset_name)
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for x_data_split in np.array_split(x_data, 4):
            executor.submit(__write_noise_dataset, x_data_split, dataset_name, noise_amount)

    finish = time.perf_counter()
    print(f'Noisy dataset successfully generated in {round(finish-start, 2)} seconds')


