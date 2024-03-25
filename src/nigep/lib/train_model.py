from keras.models import Sequential


def train_model(model: Sequential, epochs, callbacks, train_data, verbose=0):
    x_train, y_train = train_data

    model.fit(
        x_train,
        y_train,
        verbose=verbose,
        callbacks=callbacks,
        epochs=epochs,
        validation_split=.1,
    )
