import os
from tabulate import tabulate
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import pandas as pd
def calculateMetrics(df):
  f1 = f1_score(df.Labels, df.Predictions>=.5)
  precision = precision_score(df.Labels, df.Predictions>=.5)
  recall = recall_score(df.Labels, df.Predictions>=.5)
  return [f1, precision, recall]
def create_table(train: str, val: str, test: str, names, models):
    """
    :param train: Path to train results
    :param val: Path to validation results
    :param test: Path to test results
    :param names: list of names to give to the models in the table
    :param models: list of file names within the directories given
    :return: matplotlib plot
    """
    results = np.zeros((3 * 3, len(models)))

    indexes = []
    metrics = ["F1", "Recall", "Precision"]
    datasets = ["Train", "Validation", "Test"]
    for dataset in datasets:
      for metric in metrics:
        indexes.append(dataset+" "+metric)
    for i, (name, file) in enumerate(zip(names, models)):
      results[0:3, i] = calculateMetrics(pd.read_csv(train+file+".csv"))
      results[3:6, i] = calculateMetrics(pd.read_csv(val+file+".csv"))
      results[6:9, i] = calculateMetrics(pd.read_csv(test+file+".csv"))


    table = tabulate(results, headers=names, tablefmt="fancy_grid", showindex=indexes)

    return table

if __name__ == '__main__':
    table = create_table("./Results/train/", "./Results/validation/", "./Results/test/", ["Bert", "Bert Improved"], ["Bert", "Bert Improved"])
    print(table)