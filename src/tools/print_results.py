from typing import List
import os
import sys

import pandas as pd
import argparse

parser = argparse.ArgumentParser()
# Model-unspecific commandline arguments
parser.add_argument('-m', '--model',
                    required=True,
                    type=str,
                    choices=['CNN', 'BiLSTM', 'BiGRU', 'CLF'],
                    help='Select a predefined model.')
parser.add_argument('-fp', '--file_path',
                    required=True,
                    type=str,
                    help='path to file without root path')
parser.add_argument('-p', '--parameters',
                    required=True,
                    nargs='+',
                    help='list of parameters',)
parser.add_argument('-me', '--metrics',
                    required=False,
                    nargs='+',
                    help='list of metrics',)
parser.add_argument('-s', '--statistics',
                    required=False,
                    nargs='+',
                    help='list of statistics (mean/std)',)


try:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    root = os.path.dirname(os.getcwd())
sys.path.append(root)
#print('Project root: {}'.format(root))

args = parser.parse_args()


def print_res(file_path: str,
              model: str,
              parameters: List[str],
              metrics: List[str] = ['f1_macro_sk', 'f1_micro_sk'],
              statistics: List[str] = ['mean', 'std']):

    full_path = os.path.join(root, file_path)
    df = pd.read_excel(full_path)
    df_sub = df[df['Model'] == model].groupby(parameters)[metrics].agg(statistics).reset_index()

    print(df_sub)


if __name__ == '__main__':
    path = args.file_path
    model = args.model
    parameters = args.parameters
    statistics = args.statistics
    metrics = args.metrics

    if metrics is not None:
        print_res(path, model, parameters, metrics)
    else:
        print_res(path, model, parameters)
