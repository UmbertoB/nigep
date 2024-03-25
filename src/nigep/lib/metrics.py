from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


def get_model_predictions(model, x_test):
    predict_x = model.predict(x_test)
    return np.argmax(predict_x, axis=-1)


def get_confusion_matrix_and_report(y_test, predictions, target_names):
    y_true = np.argmax(y_test, axis=-1)
    cm = confusion_matrix(y_true, predictions)
    cr = classification_report(y_true, predictions, target_names=target_names, labels=np.arange(0,len(target_names),1))
    return cm, cr


def compute_metrics(model, target_names, x_test, y_test):
    predictions = get_model_predictions(model, x_test)
    cm, cr = get_confusion_matrix_and_report(y_test, predictions, target_names)
    return cm, cr
