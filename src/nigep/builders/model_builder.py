from keras.models import Sequential


def train_model_for_dataset(model: Sequential, train_data, epochs, callbacks):
    x_train, y_train = train_data

    if callbacks is None:
        callbacks = []

    history = model.fit(
        x_train,
        y_train,
        callbacks=callbacks,
        epochs=epochs
    )

    return history
