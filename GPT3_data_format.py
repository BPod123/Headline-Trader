import os
from data import load_data, load_data_memory_efficient
import html
import re
import numpy as np
from collections import Counter
from BOW_data_format import clean_string

def to_input_label_list(dataset):
    titles = list(map(clean_string, (' '.join(x[0]['title']) for x in dataset)))
    labels = [int(x[1]['Label']) for x in dataset]
    return titles, labels

def prepare_data():
    db_path = "C:\\Users\\jakob\\Documents\\Headline-Trader\\Data\\Headlines.db"
    real_news, [train_stock, valid_stock, test_stock] = load_data_memory_efficient(db_path, 5, (0.8, 0.1, 0.1), False, True, ["SPY", "^DJI", "NDAQ", "AAPL", "GOOG", "META"])
    title_gen = list(map(lambda x: (x[0], (clean_string(' '.join(x[1])))), real_news[['date_int', 'title']].reset_index().drop(columns=['index']).groupby('date_int')['title']))
    titles = []
    for idx, tokens in title_gen:
        if len(titles) < idx:
            titles.extend([None for i in range(idx - len(titles))])
        titles.append(tokens)

    title_data = {idx: titles[idx] for idx in range(len(titles)) if titles[idx] is not None}
    stock_format = lambda stock: np.ascontiguousarray(np.vstack([stock['date_int'].to_numpy(), stock['Label'].to_numpy()]).T.astype(int))
    return {
        'title_data': title_data,
        'train': stock_format(train_stock),
        'validation': stock_format(valid_stock),
        'test': stock_format(test_stock),
    }


