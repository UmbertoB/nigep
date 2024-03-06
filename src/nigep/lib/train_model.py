from keras.models import Sequential


def train_model(model: Sequential, epochs, callbacks, train_data):
    x_train, y_train = train_data

    model.fit(
        x_train,
        y_train,
        verbose=0,
        callbacks=callbacks,
        epochs=epochs,
        validation_split=.1,
    )

