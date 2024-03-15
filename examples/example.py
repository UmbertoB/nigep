# imports


import os
import sys

module_path = os.path.abspath(os.path.join('../src/nigep/*'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.nigep import Nigep
import keras.layers as layers
import keras.applications as applications
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.optimizers.legacy import SGD
from keras.datasets import cifar10
from keras.utils import to_categorical

# DATASET

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255
x_test = x_test / 255
OUTPUT_DIMS = 10
y_train = to_categorical(y_train, OUTPUT_DIMS)
y_test = to_categorical(y_test, OUTPUT_DIMS)

# Model

base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

model = Sequential(layers=[
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(10, activation='softmax')
])

model.summary()

model.compile(optimizer=SGD(learning_rate=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# EXECUTE

erl_stopping = EarlyStopping(patience=4, monitor='val_loss', verbose=1)
callbacks = [erl_stopping]

nigep = Nigep(
    execution_name='cifar10',
    x_data=x_train,
    y_data=y_train,
    model=model,
    batch_size=64,
    input_shape=(32, 32),
    class_mode='categorical',
    k_fold_n=5,
    target_names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    epochs=20,
    callbacks=callbacks
)

nigep.execute()
