from keras.models import Sequential


def train_model(model: Sequential, epochs, callbacks, train_data, verbose=0):

    model.fit(
        train_data,
        callbacks=callbacks,
        epochs=epochs,
    )
