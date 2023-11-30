from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


def get_model_predictions(model, test_generator, class_mode):
    predict_x = model.predict(test_generator)

    if class_mode is 'binary':
        return np.round(predict_x).flatten().astype(int)

    return np.argmax(predict_x, axis=-1)


def get_confusion_matrix_and_report(test_generator, predictions, target_names):
    cm = confusion_matrix(test_generator.classes, predictions)
    cr = classification_report(test_generator.classes, predictions, target_names=target_names, labels=np.arange(0,len(target_names),1))
    return cm, cr
