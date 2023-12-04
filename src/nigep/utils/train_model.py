from keras.models import Sequential


def train_model_for_dataset(train_data, model: Sequential, epochs, callbacks):
    x_train, y_train = train_data

    if callbacks is None:
        callbacks = []

    model.fit(
        x_train,
        y_train,
        callbacks=callbacks,
        epochs=epochs
    )
