import os
import traceback

from keras.models import Sequential
from sklearn.model_selection import KFold

from .builders import model_builder
from .builders.dataset_builder import generate_dataset
from .builders.image_generator_builder import get_train_generator, get_test_generator
from .builders.metrics_builder import get_confusion_matrix_and_report, get_model_predictions
from .utils.consts import NOISE_LEVELS
from .utils.results_writer import ResultsWriter
from .utils.functions import noise_datasets_already_exists


class Nigep:

    def __init__(self,
                 execution_name: str,
                 model: Sequential,
                 batch_size: int,
                 input_shape,
                 x_data,
                 y_data,
                 target_names=None,
                 class_mode='categorical',
                 k_fold_n=5,
                 epochs=10,
                 callbacks=None,
                 ):
        if callbacks is None:
            callbacks = []
        self.execution_name = execution_name
        self.model = model
        self.batch_size = batch_size
        self.x_data = x_data
        self.y_data = y_data
        self.input_shape = input_shape
        self.class_mode = class_mode
        self.k_fold_n = k_fold_n
        self.target_names = target_names
        self.epochs = epochs
        self.callbacks = callbacks

    def __generate_noisy_datasets(self):
        for noise_amount in NOISE_LEVELS:
            dataset_name = f'noise_{noise_amount}'
            generate_dataset(self.x_data, dataset_name, noise_amount)

    def execute(self):
        rw = ResultsWriter(self.execution_name)

        if not noise_datasets_already_exists():
            self.__generate_noisy_datasets()

        kf = KFold(n_splits=self.k_fold_n, shuffle=True, random_state=42)

        for train_index, test_index in kf.split(self.x_data, self.y_data):
            rw.write_execution_folder()

            try:
                for noise_amount in NOISE_LEVELS:

                    train_gen, val_gen = get_train_generator(self.x_data, self.y_data, noise_amount, train_index,
                                                             self.batch_size, self.class_mode, self.input_shape)

                    model_builder.train_model_for_dataset(self.model, self.epochs, self.callbacks, train_gen, val_gen)

                    rw.write_model(self.model, f'train_{noise_amount}')

                    for noise_amount_testing in NOISE_LEVELS:
                        test_gen = get_test_generator(self.x_data, self.y_data, noise_amount_testing, test_index,
                                                      self.batch_size, self.class_mode, self.input_shape)
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

            except Exception as e:
                print(e)
                traceback.print_tb(e.__traceback__)
                rw.delete_results()

        rw.generate_mean_csv()
