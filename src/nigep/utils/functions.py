import os


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def get_results_columns(target_names):
    classes_precision_columns = []
    classes_recall_columns = []
    classes_f1score_columns = []
    for index, item in enumerate(target_names):
        classes_precision_columns.append(f'precision({item})')
        classes_recall_columns.append(f'recall({item})')
        classes_f1score_columns.append(f'f1-score({item})')

    return [
        'train-dataset-noise', 'test-dataset-noise',
        *classes_precision_columns,
        'precision(macro-avg)', 'precision(weighted-avg)',
        *classes_recall_columns,
        'recall(macro-avg)',  'recall(weighted-avg)',
        *classes_f1score_columns,
        'f1-score(accuracy)', 'f1-score(macro-avg)', 'f1-score(weighted-avg)'
    ]


def noisy_datasets_already_exists(noise_levels):
    for noise_level in noise_levels:
        folder_name = f"noise_{noise_level}"
        folder_path = os.path.join(f'{os.getcwd()}/dataset', folder_name)

        if not os.path.exists(folder_path):
            return False

    return True
