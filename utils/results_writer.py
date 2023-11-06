import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils.consts import results_columns
from utils.functions import isfloat
from utils.mkdir_folders import mkdir_output


class ResultsWriter:

    def __init__(self, name):
        mkdir_output()
        self.execution_folder_path = f'{os.getcwd()}/output/{name}'
        os.mkdir(self.execution_folder_path)
        self.results_folder = '/'
        self.results_name = name

    def __generate_df_by_csv(self):
        df = pd.read_csv(self.results_folder + f'/results_{self.results_name}.csv')
        df.drop('Unnamed: 0', axis=1, inplace=True)
        return df

    def __generate_results_csv(self):
        pd.DataFrame(columns=results_columns).to_csv(self.results_folder + f'/results_{self.results_name}.csv')

    def write_metrics_results(self, model_name, train_dataset_noise, test_dataset_noise, cr, cm):
        if not os.path.isfile(self.results_folder + f'/results_{self.results_name}.csv'):
            self.__generate_results_csv()

        df = self.__generate_df_by_csv()
        split_string = [x.split(' ') for x in cr.split('\n')]
        only_values_list = []
        for row in split_string:
            only_values_list.append(list(filter((lambda x: isfloat(x)), row)))
        values = list(filter((lambda x: len(x) > 0), only_values_list))

        sequence = [[
            model_name, train_dataset_noise, test_dataset_noise, values[0][0], values[1][0], values[3][0],
            values[4][0], values[0][1], values[1][1], values[3][1], values[4][1],
            values[0][2], values[1][2], values[2][0], values[3][2], values[4][2],
            cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        ]]

        pd.concat([df, pd.DataFrame(data=sequence, columns=df.columns)], ignore_index=True) \
            .to_csv(self.results_folder + f'/results_{self.results_name}.csv')

    def write_execution_folder(self):
        self.results_folder = f'{self.execution_folder_path}/results_{self.results_name}_{datetime.now().isoformat().__str__()}'
        os.mkdir(self.results_folder)

    def write_model(self, model, model_name):
        model.save(f'{self.results_folder}/{model_name}.hdf5')

    def delete_results(self):
        os.rmdir(self.results_folder)

    def generate_mean_csv(self):
        sns.set_theme(style="white")
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

        mean_df.to_csv(
            f'{os.getcwd()}/output/{self.results_name}/mean_results.csv')

        column_values = mean_df['train-noise'].unique()
        conf_matrix = []

        for value in column_values:
            value_df = mean_df[mean_df['train-noise'] == value]
            conf_matrix.append(value_df['f1-score(weighted-avg)'].to_numpy())

        heatmap_df = pd.DataFrame(conf_matrix, columns=column_values, index=column_values)

        sns.set_theme(style="whitegrid")
        sns.set(font_scale=2)
        sns.set_context("paper")
        sns.set(font='serif')
        sns.set_style("white", {
            "font.family": "serif",
            "font.serif": ["Times", "Palatino", "serif"]
        })

        plt.figure(figsize=(8, 6))
        sns.heatmap(heatmap_df,
                    cmap='coolwarm',
                    annot=True,
                    cbar_kws={'label': '5-Fold F-Score mean vs. 0 Test Noise'},
                    fmt='.2f')

        plt.xlabel('Test Noise')
        plt.ylabel('Train Noise')

        plt.savefig(f'{os.getcwd()}/output/{self.results_name}/mean_results_heatmap.png')
