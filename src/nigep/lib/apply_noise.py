import numpy as np
from keras.preprocessing.image import ImageDataGenerator


def apply_noise(x_data, y_data, index, noise_amount):
    xx_data, yy_data = x_data[index], y_data[index]

    for idx, img in enumerate(xx_data):
        img_size = img.size
        noise_percentage = noise_amount
        noise_size = int(noise_percentage * img_size)
        random_indices = np.random.choice(img_size, noise_size)
        img_noised = img.copy()
        noise = np.random.choice([img.min(), img.max()], noise_size)
        img_noised.flat[random_indices] = noise
        xx_data[idx] = img_noised

    return xx_data, yy_data


def apply_speckle_noise(x_data, y_data, batch_size, noise_level):

    def speckle(image, sigma):
        gauss = np.random.normal(0, sigma, image.shape)
        noisy = image + image * gauss
        return np.clip(noisy, 0, 1)

    def speckle_augmentation(sigma):
        return lambda x: speckle(x, sigma)

    datagen = ImageDataGenerator(
        preprocessing_function=speckle_augmentation(sigma=noise_level)
    )

    generator = datagen.flow(x_data, y_data, batch_size)

    return generator

