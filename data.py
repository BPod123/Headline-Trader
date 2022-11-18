import sqlite3
import pandas as pd
import datetime
import yfinance
import numpy as np
from torch.utils.data import Dataset
import datasets
import os
import pickle

START_DATE = datetime.date(2022, 3, 7)
def date_to_int(string):
    string = string.split(' ')[0]
    date = datetime.datetime.strptime(string, '%Y-%m-%d').date()
    delta = date - START_DATE
    return str(delta.days)
def load_data_from_pickle(database_path: str, shift=5, percentage_split=(0.8, 0.1, 0.1), gap=False, pull_from_first_section=False):
    # check if pickle exists if it doesn't then create it
    if not os.path.exists("data.pickle"):
        ds = load_data(database_path, shift=5, percentage_split=(0.8, 0.1, 0.1), gap=False, pull_from_first_section=False)
        with open("data.pickle", "wb") as f:
            pickle.dump(ds, f)
        return ds
    else:
        with open("data.pickle", "rb") as f:
            ds = pickle.load(f)
    return ds

def load_data(database_path: str, shift=5, percentage_split=(0.8, 0.1, 0.1), gap=False, pull_from_first_section=False):
    """
    :param database_path: Path to headlines database
    :param shift: The number of ticks offset the data and labels are. A shift of 5 is the same as one week as weekends
        are clipped from the data.
    :param percentage_split: iterable floats that sum to 1 that represent how the data is split up chronologically
    :param gap: If true, will include a gap where the labels of one section of data overlap with the headlines of the
        next section.
    :param pull_from_first_section: If gap is False, this will do nothing. If True and gap is True, the first
        section of data will be shrunk to allow the other sections to maintain their percentages. If False,
        then each section before the last will be shortened by "shift" to accommodate the gap for
        the proceeding section.
    :return: array of length len(percentage_split)
    """
    con = sqlite3.connect(database_path)
    con.create_function("DATE_TO_INT", 1, date_to_int)
    df = pd.read_sql("""
    SELECT DATE_TO_INT(date) as 'date_int', name, date, title, description
    FROM headline JOIN feed on feed.url=headline.url
    WHERE CAST(date_int as int) >= 0 and
     (CAST(date_int as int)) % 7 < 5 """, con)
    con.close()
    df['date_int'] = pd.to_numeric(df['date_int'])
    df['date'] = pd.to_datetime(df['date']).apply(datetime.datetime.date)
    real_news = df[(df['name'] != 'Babylon Bee (Fake News)') & (df['name'] != 'The Onion (Fake News)')].sort_values('date_int').reset_index().drop(columns=['index'])


    end_date = START_DATE + datetime.timedelta(days=max(real_news['date_int']))
    data = yfinance.download("SPY", start=str(START_DATE), end=str(end_date))['Close'].reset_index()
    data['Date'] = data['Date'].apply(datetime.datetime.date)
    stock_dates = set(data['Date'])
    days = [x for x in [real_news[real_news['date_int'] == x].reset_index().drop(columns=['index']) for x in
                        sorted(set(real_news['date_int']))] if x['date'][0] in stock_dates]

    # Add shift spaces to the end to use for padding
    closing_prices = np.concatenate([data['Close'].to_numpy(), np.zeros((shift,))])

    stacked = np.stack([np.roll(closing_prices, shift), closing_prices])
    data['Label'] = (stacked[1] > stacked[0])[shift:]
    data = data[:-shift]
    days = days[shift:]
    all_data = list(zip(days, (data.T[i] for i in range(len(data)))))
    length = len(all_data)
    lengths = list(map(lambda x: round(x * length), percentage_split))[:-1]

    if len(percentage_split) > 1:
        start = 0
        return_data = []
        if gap and pull_from_first_section:
            lengths[0] -= shift * len(lengths)
        for subset_length in lengths:
            if gap and not pull_from_first_section:
                subset_length -= shift
            return_data.append(all_data[start:start + subset_length])
            start += subset_length
            if gap:
                start += shift
        return_data.append(all_data[start:])
        return return_data
    return all_data



if __name__ == '__main__':
    with_gaps_pull_from_first = load_data("Data/Headlines.db", 5, (0.6, 0.1, 0.3), True, True)
    with_gaps_pull_from_preceeding = load_data("Data/Headlines.db", 5, (0.6, 0.1, 0.3), True, False)
    no_gaps = load_data("Data/Headlines.db", 5, (0.6, 0.1, 0.3))


