import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .functions import get_results_columns
from .mkdir_folders import mkdir_output, get_directory_name

sns.set_theme(style="white")
sns.set_theme(style="whitegrid")
sns.set(font_scale=2)
sns.set_context("paper")
sns.set(font='serif')
sns.set_style("white", {
    "font.family": "serif",
    "font.serif": ["Times", "Palatino", "serif"]
})


class ResultsWriter:

    def __init__(self, name):
        mkdir_output()
        self.execution_folder_path = get_directory_name(f'{os.getcwd()}/output/{name}')
        os.mkdir(self.execution_folder_path)
        self.results_folder = '/'
        self.results_name = name

    def __generate_df_by_csv(self):
        df = pd.read_csv(self.results_folder + f'/results_{self.results_name}.csv')
        df.drop('Unnamed: 0', axis=1, inplace=True)
        return df

    def __generate_results_csv(self, target_names):
        pd.DataFrame(columns=get_results_columns(target_names)).to_csv(self.results_folder + f'/results_{self.results_name}.csv')

    def write_metrics_results(self, train_noise, test_noise, cr, cm, target_names):
        if not os.path.isfile(self.results_folder + f'/results_{self.results_name}.csv'):
            self.__generate_results_csv(target_names)

        current_df = self.__generate_df_by_csv()

        pattern = re.compile(r'(\w+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)')
        matches = pattern.findall(cr)

        columns = ['Class', 'Precision', 'Recall', 'F1-Score', 'Support']
        df = pd.DataFrame(matches, columns=columns).astype(
            {'Precision': float, 'Recall': float, 'F1-Score': float, 'Support': int})

        accuracy_match = re.search(r'accuracy\s+([\d.]+)', cr)
        accuracy = float(accuracy_match.group(1)) if accuracy_match else None

        classes_precision_dict = {}
        classes_recall_dict = {}
        classes_f1score_dict = {}
        for index, item in enumerate(target_names):
            classes_precision_dict[f'precision({item})'] = df['Precision'][index]
            classes_recall_dict[f'recall({item})'] = df['Recall'][index]
            classes_f1score_dict[f'f1-score({item})'] = df['F1-Score'][index]

        classes_number = len(target_names)
        metrics = {
            'train-dataset-noise': train_noise,
            'test-dataset-noise': test_noise,
            **classes_precision_dict,
            'precision(macro-avg)': df['Precision'][classes_number],
            'precision(weighted-avg)': df['Precision'][classes_number + 1],
            **classes_recall_dict,
            'recall(macro-avg)': df['Recall'][classes_number],
            'recall(weighted-avg)': df['Recall'][classes_number + 1],
            **classes_f1score_dict,
            'f1-score(accuracy)': accuracy,
            'f1-score(macro-avg)': df['F1-Score'][classes_number],
            'f1-score(weighted-avg)': df['F1-Score'][classes_number + 1],
        }

        pd.DataFrame(cm).to_csv(
            self.results_folder + f'/train_{train_noise}_test_{test_noise}_confusion_matrix.csv'
        )
        pd.concat([current_df, pd.DataFrame(metrics, index=[0])], ignore_index=True) \
            .to_csv(self.results_folder + f'/results_{self.results_name}.csv')

    def write_execution_folder(self):
        self.results_folder = f'{self.execution_folder_path}/kfold_{datetime.now().isoformat().__str__()}'
        os.mkdir(self.results_folder)

    def write_model(self, model, noise):
        model.save(f'{self.results_folder}/train_{noise}.keras')

    def delete_results(self):
        os.rmdir(self.results_folder)

    def generate_mean_csv(self):
        unified_data = []
        train_noise = []
        test_noise = []
        for subdir, _, files in os.walk(f'{os.getcwd()}/output/{self.results_name}'):
            for file in files:
                if file.endswith('.csv') and file.startswith('results'):
                    file_path = os.path.join(subdir, file)
                    csv = pd.read_csv(file_path)
                    unified_data.append(csv['f1-score(weighted-avg)'])
                    train_noise = csv['train-dataset-noise']
                    test_noise = csv['test-dataset-noise']

        merged_df = pd.concat(unified_data, axis=1)
        mean_data = {'train-noise': train_noise, 'test-noise': test_noise,
                     'f1-score(weighted-avg)': merged_df.mean(axis=1)}

        mean_df = pd.DataFrame(mean_data)

        column_values = mean_df['train-noise'].unique()
        conf_matrix = [mean_df.loc[mean_df['train-noise'] == value, 'f1-score(weighted-avg)'].to_numpy()
                       for value in column_values]

        heatmap_df = pd.DataFrame(conf_matrix, columns=column_values, index=column_values)

        mean_df.to_csv(f'{os.getcwd()}/output/{self.results_name}/mean_results.csv')
        heatmap_df.to_csv(f'{os.getcwd()}/output/{self.results_name}/generalization_profile_heatmap.csv')

        # plt.figure(figsize=(8, 6))
        # sns.heatmap(heatmap_df,
        #             cmap='coolwarm',
        #             annot=True,
        #             cbar_kws={'label': '5-Fold F-Score mean vs. 0 Test Noise'},
        #             fmt='.2f')
        #
        # plt.xlabel('Test Noise')
        # plt.ylabel('Train Noise')
        #
        # plt.savefig(f'{os.getcwd()}/output/{self.results_name}/mean_results_heatmap.png')
