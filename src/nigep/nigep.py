from keras.models import Sequential
from sklearn.model_selection import KFold

from src.nigep.utils.apply_noise import apply_noise
from src.nigep.utils.metrics import get_confusion_matrix_and_report, get_model_predictions
from src.nigep.utils.train_model import train_model_for_dataset
from .utils.consts import NOISE_LEVELS, NIGEP_AVAILABLE_KWARGS
from .utils.results_writer import ResultsWriter
from .utils.functions import validate_kwargs


class Nigep:

    def __init__(self, **kwargs):
        validate_kwargs(kwargs=kwargs, allowed_kwargs=NIGEP_AVAILABLE_KWARGS)
        self.execution_name: str = kwargs['execution_name']
        self.model: Sequential = kwargs['model']
        self.batch_size: int = kwargs['batch_size']
        self.input_shape: tuple[int, int] = kwargs['input_shape']
        self.x_data: list[str] = kwargs['x_data']
        self.y_data: list[str] = kwargs['y_data']
        self.target_names: list[str] = kwargs.get('target_names', None)
        self.class_mode: str = kwargs.get('class_mode', 'categorical')
        self.k_fold_n: int = kwargs.get('k_fold_n', 5)
        self.epochs: int = kwargs.get('epochs', 10)
        self.callbacks: list[any] = kwargs.get('callbacks', None)
        self.noise_levels: any = kwargs.get('noise_levels', NOISE_LEVELS)
        self.write_trained_models: bool = kwargs.get('write_trained_models', False)
        self.evaluate_trained_models: bool = kwargs.get('evaluate_trained_models', False)
        self.rw = ResultsWriter(self.execution_name)

    def __train_step(self, train_index, train_noise_amount):
        train_data = apply_noise(self.x_data, self.y_data, train_index, train_noise_amount)

        train_model_for_dataset(train_data, self.model, self.epochs, self.callbacks)

        if self.write_trained_models:
            self.rw.write_model(self.model, train_noise_amount)

    def __test_and_results_step(self, test_index, train_noise_amount, test_noise_amount):
        x_test, y_test = apply_noise(self.x_data, self.y_data, test_index, test_noise_amount)

        if self.evaluate_trained_models:
            self.model.evaluate(x_test, y_test)

        predictions = get_model_predictions(self.model, x_test, self.class_mode)
        cm, cr = get_confusion_matrix_and_report(y_test, predictions, self.target_names)

        self.rw.write_new_metrics(
            train_noise_amount,
            test_noise_amount,
            cr,
            cm,
            self.target_names
        )

    def execute(self):
        kf = KFold(n_splits=self.k_fold_n, shuffle=True, random_state=42)

        fold_number = 1
        for train_index, test_index in kf.split(self.x_data, self.y_data):
            self.rw.write_k_subset_folder(fold_number)
            fold_number += 1

            for train_noise_amount in self.noise_levels:

                self.__train_step(train_index, train_noise_amount)

                for test_noise_amount in self.noise_levels:
                    self.__test_and_results_step(test_index, train_noise_amount, test_noise_amount)

        self.rw.save_mean_merged_results()
        self.rw.save_heatmap_csv()

    def plot_and_save_generalization_profile(self):
        self.rw.plot_and_save_heatmap_png()
