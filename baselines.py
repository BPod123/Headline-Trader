import pandas as pd
import os
import numpy as np
# Just read the dates from one of the files in the
# train, validation, and test directories of the Results Directory
# Then create a new dataframe with random predictions.
if __name__ == '__main__':
    paths = [os.path.join('Results', x) for x in ['test', 'train', 'validation']]
    for path in paths:
        read_path = os.path.join(path, 'BOW.csv')
        save_path = os.path.join(path, 'Random_Baseline.csv')
        df = pd.read_csv(read_path)
        df['Predictions'] = np.random.uniform(0, 1, len(df))
        df.to_csv(save_path, index=False)