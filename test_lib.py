from keras.layers import Flatten, Dense
from keras.models import Sequential
from keras.preprocessing import image_dataset_from_directory
from keras.utils import get_file

model = Sequential()
model.add(Flatten(input_shape=(200, 200, 3)))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# opt = Adam(learning_rate=alpha)

model.build()
model.summary()
model.compile(loss='binary_crossentropy',
              # optimizer=opt,
              metrics=['acc'])

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = get_file(
    origin=dataset_url, fname="flower_photos", untar=True, cache_dir='./'
)

dataset = image_dataset_from_directory(
    data_dir, image_size=(180, 180), batch_size=64
)

# dataframe = pd.read_csv('./dataset/dataframe.csv')
# x_data = dataframe['images']
# y_data = dataframe['classes']
#
# nid = Nid('test1', model, 5, x_data, y_data, 2)
# #
# nid.execute()
