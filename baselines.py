from data import load_data_from_pickle
import random
import pandas as pd
import os
import numpy as np
# Just read the dates from one of the files in the
# train, validation, and test directories of the Results Directory
# Then create a new dataframe with random predictions.

def get_majority_class(ds):
    up_labels = ds.Labels.sum()
    return 1 if up_labels / len(ds) > 0.5 else 0


def majority_class_baselines():
    train_ds = pd.read_csv("Results/train/GPT3.csv")
    paths = [os.path.join('Results', x) for x in ['test', 'train', 'validation']]
    for path in paths:
        read_path = os.path.join(path, 'GPT3.csv')
        save_path = os.path.join(path, 'Majority_baseline.csv')
        df = pd.read_csv(read_path)
        pred = get_majority_class(train_ds)
        df['Predictions'] = np.ones(len(df)) * pred
        df.to_csv(save_path, index=False)


def random_baselines():
    paths = [os.path.join('Results', x) for x in ['test', 'train', 'validation']]
    for path in paths:
        read_path = os.path.join(path, 'BOW.csv')
        save_path = os.path.join(path, 'Random_Baseline.csv')
        df = pd.read_csv(read_path)
        df['Predictions'] = np.random.uniform(0, 1, len(df))
        df.to_csv(save_path, index=False)

if __name__ == "__main__":
    majority_class_baselines()
    # random_baselines()