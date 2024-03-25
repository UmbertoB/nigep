from sklearn.base import BaseEstimator, TransformerMixin

from ..lib.apply_noise import apply_noise
from ..lib.metrics import compute_metrics
from ..lib.train_model import train_model
from ..lib.functions import write_model


class NigepModel(BaseEstimator, TransformerMixin):
    def __init__(self,
                 results_folder,
                 fold_number,
                 train_noise,
                 train_index,
                 model,
                 epochs,
                 callbacks,
                 class_mode,
                 target_names,
                 save_models,
                 evaluate_models=True):
        self.results_folder = results_folder
        self.train_index = train_index
        self.train_noise = train_noise
        self.model = model
        self.fold_number = fold_number
        self.epochs = epochs
        self.callbacks = callbacks
        self.class_mode = class_mode
        self.target_names = target_names
        self.evaluate_models = evaluate_models
        self.save_models = save_models

    def fit(self, X, y=None):
        noised_data = apply_noise(X, y, self.train_index, self.train_noise)

        print(f'Fold: {str(self.fold_number)} - Training with Noise: {str(self.train_noise)}')

        train_model(self.model, self.epochs, self.callbacks, noised_data)
        write_model(self.results_folder, self.save_models, self.model, self.train_noise)

        return self

    def evaluate_model(self, X, y, test_index, test_noise):
        print(f'Fold: {str(self.fold_number)} - Train Noise: {str(self.train_noise)} - Test Noise: {str(test_noise)}')

        x_test, y_test = apply_noise(X, y, test_index, test_noise)

        if self.evaluate_models:
            self.model.evaluate(x_test, y_test)

        cm, cr = compute_metrics(self.model, self.class_mode, self.target_names, x_test, y_test)

        return cm, cr

