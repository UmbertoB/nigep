from keras.models import Sequential


def train_model_for_dataset(model: Sequential, epochs, callbacks, train_generator, validation_generator):
    if callbacks is None:
        callbacks = []

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        callbacks=callbacks,
        epochs=epochs
    )

    return history
