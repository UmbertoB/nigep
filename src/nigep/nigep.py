from keras.models import Sequential
from sklearn.model_selection import KFold

from .builders import model_builder
from .builders.image_generator_builder import get_train_data, get_test_data
from .builders.metrics_builder import get_confusion_matrix_and_report, get_model_predictions
from .utils.consts import NOISE_LEVELS, NIGEP_AVAILABLE_KWARGS
from .utils.results_writer import ResultsWriter
from .utils.functions import validate_kwargs
from .layers.salt_and_pepper_noise import SaltAndPepperNoise


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

    def execute(self):
        rw = ResultsWriter(self.execution_name)
        kf = KFold(n_splits=self.k_fold_n, shuffle=True, random_state=42)

        for train_index, test_index in kf.split(self.x_data, self.y_data):
            rw.write_execution_folder()

            for noise_amount in self.noise_levels:

                train_data = get_train_data(self.x_data, self.y_data, train_index)

                nigep_model = Sequential()
                nigep_model.add(SaltAndPepperNoise(noise_amount, input_shape=(self.input_shape,)))
                nigep_model.add(self.model)

                nigep_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

                model_builder.train_model_for_dataset(nigep_model,
                                                      train_data,
                                                      self.epochs,
                                                      self.callbacks)

                if self.write_trained_models:
                    rw.write_model(nigep_model, noise_amount)

                for noise_amount_testing in self.noise_levels:
                    x_test, y_test = get_test_data(self.x_data, self.y_data, test_index, noise_amount_testing)

                    if self.evaluate_trained_models:
                        nigep_model.evaluate(x_test, y_test)

                    predictions = get_model_predictions(nigep_model, x_test, self.class_mode)
                    cm, cr = get_confusion_matrix_and_report(y_test, predictions, self.target_names)

                    rw.write_metrics_results(
                        noise_amount,
                        noise_amount_testing,
                        cr,
                        cm,
                        self.target_names
                    )

        rw.generate_mean_csv()
