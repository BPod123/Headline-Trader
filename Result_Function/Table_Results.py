import os
from tabulate import tabulate
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import pandas as pd


def calculateMetrics(df):

    f1 = f1_score(df.Labels, df.Predictions >= .5)
    precision = precision_score(df.Labels, df.Predictions >= .5)
    recall = recall_score(df.Labels, df.Predictions >= .5)

    return [f1, precision, recall]


def create_table(train: str, val: str, test: str, models):
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

    for i, file in enumerate(models):
        results[0:3, i] = calculateMetrics(pd.read_csv(train+file+".csv"))
        results[3:6, i] = calculateMetrics(pd.read_csv(val+file+".csv"))
        results[6:9, i] = calculateMetrics(pd.read_csv(test+file+".csv"))

    # print("results")
    # print(results)
    # print("names")
    # print(names)
    # print("indexes")
    # print(indexes)

    print("LaTeX Code:")

    # header
    print("   ", end="")
    for model in models:
        print(f" & {model}", end="")
    print(r" \\")
    print(r"\hline")

    # results
    for row in range(len(indexes)):

        # Row Name
        print(f"{indexes[row]}", end="")

        # Results for the Row
        for j in range(len(models)):
            print(f" & {round(results[row, j], 3)}", end="")

        print(r" \\")
        print(r"\hline")


    table = tabulate(results, headers=models, tablefmt="fancy_grid", showindex=indexes)

    return table

if __name__ == '__main__':
    table = create_table("../Results/train/", "../Results/validation/", "../Results/test/",
                         ["Majority_baseline", "Random_Baseline", "BOW", "Bert", "Bert Improved", "GPT3"])
    print(table)