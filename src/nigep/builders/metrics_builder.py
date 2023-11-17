from sklearn.metrics import classification_report, confusion_matrix


def get_model_predictions(model, test_generator):
    predictions = model.predict(test_generator)
    all_predictions = []
    for pred in predictions:
        all_predictions.append((pred[0] > 0.5).astype("int32"))

    return all_predictions


def get_confusion_matrix_and_report(test_generator, predictions, target_names):
    cm = confusion_matrix(test_generator.classes, predictions)
    cr = classification_report(test_generator.classes, predictions, target_names=target_names)
    return cm, cr
