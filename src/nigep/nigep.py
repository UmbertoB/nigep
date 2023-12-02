import traceback

from keras.models import Sequential
from sklearn.model_selection import KFold

from .builders import model_builder
from .builders.dataset_builder import generate_dataset
from .builders.image_generator_builder import get_train_generator, get_test_generator
from .builders.metrics_builder import get_confusion_matrix_and_report, get_model_predictions
from .utils.consts import NOISE_LEVELS, NIGEP_AVAILABLE_KWARGS
from .utils.results_writer import ResultsWriter
from .utils.functions import noisy_datasets_already_exists, validate_kwargs


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

    def __generate_noisy_datasets(self):
        for noise_amount in self.noise_levels:
            dataset_name = f'noise_{noise_amount}'
            generate_dataset(self.x_data, dataset_name, noise_amount)

    def __train_and_evaluate(self):
        rw = ResultsWriter(self.execution_name)
        kf = KFold(n_splits=self.k_fold_n, shuffle=True, random_state=42)

        for train_index, test_index in kf.split(self.x_data, self.y_data):
            rw.write_execution_folder()

            for noise_amount in self.noise_levels:

                train_gen, val_gen = get_train_generator(self.x_data,
                                                         self.y_data,
                                                         self.batch_size,
                                                         self.class_mode,
                                                         self.input_shape,
                                                         noise_amount,
                                                         train_index)

                model_builder.train_model_for_dataset(self.model,
                                                      self.epochs,
                                                      self.callbacks,
                                                      train_gen,
                                                      val_gen)

                if self.write_trained_models:
                    rw.write_model(self.model, noise_amount)

                for noise_amount_testing in self.noise_levels:
                    test_gen = get_test_generator(self.x_data, self.y_data, self.batch_size, self.class_mode,
                                                  self.input_shape, noise_amount_testing, test_index)

                    if self.evaluate_trained_models:
                        self.model.evaluate(test_gen)

                    predictions = get_model_predictions(self.model, test_gen, self.class_mode)
                    cm, cr = get_confusion_matrix_and_report(test_gen, predictions, self.target_names)

                    rw.write_metrics_results(
                        noise_amount,
                        noise_amount_testing,
                        cr,
                        cm,
                        self.target_names
                    )

            rw.generate_mean_csv()

    def execute(self):
        if not noisy_datasets_already_exists(self.noise_levels):
            self.__generate_noisy_datasets()

        self.__train_and_evaluate()

