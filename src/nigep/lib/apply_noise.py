import numpy as np


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


def speckle_noise(image, sigma):
    row, col, _ = image.shape
    gauss = np.random.normal(0, sigma, (row, col, 3))
    noisy = image + image * gauss
    noisy = np.clip(noisy, 0, 1)
    return noisy

