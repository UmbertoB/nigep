import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


def get_train_data(x_data, y_data, train_index):
    x_train, y_train = x_data[train_index], y_data[train_index]

    return x_train, y_train


def get_test_data(x_data, y_data, test_index, noise_amount):
    x_test, y_test = x_data[test_index], y_data[test_index]

    for idx, img in enumerate(x_test):
        img_size = img.size
        noise_percentage = noise_amount  # Setting to 10%
        noise_size = int(noise_percentage * img_size)
        random_indices = np.random.choice(img_size, noise_size)
        img_noised = img.copy()
        noise = np.random.choice([img.min(), img.max()], noise_size)
        img_noised.flat[random_indices] = noise
        x_test[idx] = img_noised

    return x_test, y_test
