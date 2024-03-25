def get_results_columns():
    return [
        'fold_number',
        'precision(macro-avg)',
        'precision(weighted-avg)',
        'recall(macro-avg)',
        'recall(weighted-avg)',
        'f1-score(accuracy)',
        'f1-score(macro-avg)',
        'f1-score(weighted-avg)'
    ]


def validate_kwargs(
    kwargs, allowed_kwargs, error_message="Keyword argument not understood:"
):
    """Checks that all keyword arguments are in the set of allowed keys."""
    for kwarg in kwargs:
        if kwarg not in allowed_kwargs:
            raise TypeError(error_message, kwarg)


def write_model(results_folder, save_model, model):
    if save_model:
        model.save(f'{results_folder}/model.keras')

