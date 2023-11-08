from keras.models import Sequential


# def __create_callbacks(alpha):
#     lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_delta=alpha, patience=5, verbose=1)
#     erl_stopping = tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss', verbose=1)
#     callbacks = [lr_reduce, erl_stopping]
#     return callbacks

def train_model_for_dataset(model: Sequential, train_generator, validation_generator):
    # callbacks = __create_callbacks(alpha)

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        # callbacks=callbacks,
        epochs=1
    )

    return history
