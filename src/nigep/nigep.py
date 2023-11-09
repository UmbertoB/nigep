import os
import traceback

from keras.models import Sequential
from sklearn.model_selection import KFold

from .builders import model_builder
from .builders.dataset_builder import generate_dataset
from .builders.image_generator_builder import get_train_generator, get_test_generator
from .builders.metrics_builder import generate_confusion_matrix, generate_classification_report
from .utils.consts import NOISE_LEVELS
from .utils.results_writer import ResultsWriter


def noise_datasets_already_exists():
    noise_levels = [0] + [i / 10 for i in range(1, 10)]
    for noise_level in noise_levels:
        folder_name = f"noise_{noise_level}"
        folder_path = os.path.join(f'{os.getcwd()}/dataset', folder_name)

        if not os.path.exists(folder_path):
            return False

    return True


class Nigep:

    def __init__(self, execution_name: str, model: Sequential, batch_size: int, x_data, y_data, k_fold_n=5):
        self.execution_name = execution_name
        self.model = model
        self.batch_size = batch_size
        self.x_data = x_data
        self.y_data = y_data
        self.k_fold_n = k_fold_n

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

                    train_gen, val_gen = get_train_generator(self.x_data, self.y_data, noise_amount, train_index)

                    model_builder.train_model_for_dataset(self.model, train_gen, val_gen)

                    rw.write_model(self.model, f'train_{noise_amount}')

                    for noise_amount_testing in NOISE_LEVELS:
                        test_gen = get_test_generator(self.x_data, self.y_data, noise_amount_testing, test_index)
                        self.model.evaluate(test_gen)

                        cm = generate_confusion_matrix(self.model, test_gen, self.batch_size)
                        cr = generate_classification_report(self.model, test_gen, self.batch_size)

                        rw.write_metrics_results(
                            f'train_{noise_amount}_test_{noise_amount_testing}',
                            noise_amount,
                            noise_amount_testing,
                            cr,
                            cm
                        )

            except Exception as e:
                print(e)
                traceback.print_tb(e.__traceback__)
                rw.delete_results()

        rw.generate_mean_csv()
