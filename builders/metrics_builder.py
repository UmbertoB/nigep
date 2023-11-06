from sklearn.metrics import classification_report, confusion_matrix


def __get_model_predictions(model, test_generator, batch_size):
    predictions = model.predict(test_generator, 300 // batch_size + 1)
    all_predictions = []
    for pred in predictions:
        all_predictions.append((pred[0] > 0.5).astype("int32"))

    return all_predictions


def __get_confusion_matrix(model, test_generator, batch_size):
    predictions = __get_model_predictions(model, test_generator, batch_size)
    return confusion_matrix(test_generator.classes, predictions)


def __get_classification_report(model, test_generator, batch_size):
    predictions = __get_model_predictions(model, test_generator, batch_size)
    target_names = ['yes', 'no']
    return classification_report(test_generator.classes, predictions, target_names=target_names)


def generate_confusion_matrix(model, test_generator, batch_size):
    cm = __get_confusion_matrix(model, test_generator, batch_size)
    return cm


def generate_classification_report(model, test_generator, batch_size):
    cr = __get_classification_report(model, test_generator, batch_size)
    return cr
