import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


def get_x_noisy_data(noise, x_data):
    x_noisy_data = list(map(lambda p: p.replace('default', f'noise_{noise}'), x_data))
    return np.array(x_noisy_data)


def get_train_generator(x_data, y_data, batch_size, class_mode, input_shape, noise, train_index):
    x_noisy_data = get_x_noisy_data(noise, x_data)

    x_train, y_train = x_noisy_data[train_index], y_data[train_index]

    df_train = pd.DataFrame({'id': x_train, 'label': y_train})

    train_df, validation_df = train_test_split(df_train, test_size=0.1, random_state=42)

    train_gen = ImageDataGenerator(
        rotation_range=40,
        rescale=1 / 255,
        horizontal_flip=True,
        vertical_flip=True,
    )

    train_generator = train_gen.flow_from_dataframe(
        dataframe=train_df,
        x_col='id',
        y_col='label',
        batch_size=batch_size,
        target_size=input_shape,
        class_mode=class_mode,
        shuffle=True,
    )

    validation_generator = train_gen.flow_from_dataframe(
        dataframe=validation_df,
        x_col='id',
        y_col='label',
        batch_size=batch_size,
        target_size=input_shape,
        class_mode=class_mode,
        shuffle=True,
    )

    return train_generator, validation_generator


def get_test_generator(x_data, y_data, batch_size, class_mode, input_shape, noise, test_index):
    x_noisy_data = get_x_noisy_data(noise, x_data)

    x_test, y_test = x_noisy_data[test_index], y_data[test_index]

    df_test = pd.DataFrame({'id': x_test, 'label': y_test})

    test_generator = ImageDataGenerator(
        rescale=1 / 255,
    ).flow_from_dataframe(
        dataframe=df_test,
        x_col='id',
        y_col='label',
        batch_size=batch_size,
        target_size=input_shape,
        class_mode=class_mode,
        shuffle=True)

    return test_generator
