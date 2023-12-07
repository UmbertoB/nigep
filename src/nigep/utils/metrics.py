from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
import numpy as np


def get_model_predictions(model, x_test, class_mode):
    predict_x = model.predict(x_test)
    if class_mode == 'binary':
        return np.round(predict_x).flatten().astype(int)

    return np.argmax(predict_x, axis=-1)


def get_confusion_matrix_and_report(y_test, predictions, target_names):
    y_true = np.argmax(y_test, axis=-1)
    cm = confusion_matrix(y_true, predictions)
    cr = classification_report(y_true, predictions, target_names=target_names, labels=np.arange(0,len(target_names),1))
    return cm, cr


def compute_metrics(model, class_mode, target_names, x_test, y_test):
    predictions = get_model_predictions(model, x_test, class_mode)
    cm, cr = get_confusion_matrix_and_report(y_test, predictions, target_names)
    return cm, cr
