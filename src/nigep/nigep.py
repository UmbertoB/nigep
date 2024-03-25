from keras.models import Sequential
from sklearn.model_selection import KFold
from concurrent.futures import ThreadPoolExecutor
import threading

from .lib.metrics import compute_metrics
from .lib.train_model import train_model
from .lib.consts import NIGEP_AVAILABLE_KWARGS
from .classes.results_writer import ResultsWriter
from .lib.functions import validate_kwargs, write_model


class Nigep:

    def __init__(self, **kwargs):
        self.lock = threading.Lock()
        """
        Initialize Nigep instance.

        Parameters:
        - execution_name (str): Name of the execution.
        - model (Sequential): Keras Sequential model.
        - batch_size (int): Batch size for training.
        - input_shape (tuple[int, int]): Input shape of the data.
        - x_data (list[any]): Input data.
        - y_data (list[any]): Target data.
        - target_names (list[str], optional): Names of target classes.
        - class_mode (str, optional): Classification mode (default: 'categorical').
        - k_fold_n (int, optional): Number of folds in k-fold cross-validation (default: 5).
        - epochs (int, optional): Number of training epochs (default: 10).
        - callbacks (list[any], optional): List of callbacks for model training.
        - noise_levels (any, optional): Noise levels for data augmentation (default: NOISE_LEVELS).
        - save_models (bool, optional): Save trained models (default: False).
        - evaluate_models (bool, optional): Evaluate models during testing (default: False).
        - kfold_random_state (int, optional): K-Fold random state.
        """
        validate_kwargs(kwargs=kwargs, allowed_kwargs=NIGEP_AVAILABLE_KWARGS)
        self.execution_name: str = kwargs['execution_name']
        self.model: Sequential = kwargs['model']
        self.input_shape: tuple[int, int] = kwargs['input_shape']
        self.x_data: list[any] = kwargs['x_data']
        self.y_data: list[any] = kwargs['y_data']
        self.target_names: list[str] = kwargs.get('target_names', [])
        self.class_mode: str = kwargs.get('class_mode', 'categorical')
        self.k_fold_n: int = kwargs.get('k_fold_n', 5)
        self.epochs: int = kwargs.get('epochs', 10)
        self.callbacks: list[any] = kwargs.get('callbacks', [])
        self.save_models: bool = kwargs.get('save_models', False)
        self.evaluate_models: bool = kwargs.get('evaluate_models', False)
        self.kfold_random_state: int = kwargs.get('kfold_random_state', 42)
        self.rw = ResultsWriter(self.execution_name)

    def __train_and_write_model(self, results_folder, fold_number, x_train, y_train):
        print(f'Fold: {str(fold_number)} - Training')

        with self.lock:
            train_model(self.model, self.epochs, self.callbacks, (x_train, y_train))
            write_model(results_folder, self.save_models, self.model)

    def __test_and_write_metrics(self, results_folder, fold_number, x_test, y_test):
        print(f'Fold: {str(fold_number)} - Testing')

        if self.evaluate_models:
            self.model.evaluate(x_test, y_test)

        cm, cr = compute_metrics(self.model, self.class_mode, self.target_names, x_test, y_test)
        self.rw.write_new_metrics(results_folder, fold_number, cr, cm, self.target_names)

    def __execute_fold(self, fold_number, train_index, test_index):
        results_folder = self.rw.execution_folder_path

        self.__train_and_write_model(results_folder, fold_number, self.x_data[train_index], self.y_data[train_index])

        self.__test_and_write_metrics(results_folder, fold_number, self.x_data[test_index], self.y_data[test_index])

    def execute(self):
        kf = KFold(n_splits=self.k_fold_n, shuffle=True, random_state=self.kfold_random_state)
        dataset_splits = list(enumerate(kf.split(self.x_data, self.y_data)))

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(
                self.__execute_fold, fold_number, train_index, test_index)
                for fold_number, (train_index, test_index)
                in dataset_splits
            ]

            for future in futures:
                future.result()

        self.rw.save_mean_merged_results()

    def plot_and_save_generalization_profile(self):
        self.rw.plot_and_save_heatmap_png()
