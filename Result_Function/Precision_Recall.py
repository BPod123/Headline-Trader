import os
from tabulate import tabulate
from sklearn.metrics import precision_recall_curve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculateMetrics(df):
    precision_recall = precision_recall_curve(df.Labels, df.Predictions)
    return precision_recall


def create_chart(dir: str, names, models):
    """
    :param dir: Path to results
    :param names: list of names to give to the models in the table
    :param models: list of file names within the directories given
    """
    for i, (name, file) in enumerate(zip(names, models)):
        precision, recall, threshold = calculateMetrics(pd.read_csv(dir + file + ".csv"))
        plt.plot(recall, precision, label=name)

    plt.title("Recall vs. Precision")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    create_chart("../Results/test/", ["Bert", "Bert Improved", "BOW", "Random_Baseline"],["Bert", "Bert Improved", "BOW", "Random_Baseline"])
