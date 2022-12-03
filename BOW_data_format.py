import os
from data import load_data_memory_efficient
import nltk
from nltk.lm import Vocabulary
import html
import re
import numpy as np
from collections import Counter
numbers = re.compile("[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)")
contains_letters = re.compile('[a-zA-Z]')
def num_to_text(num_str):
    """
    Allows numbers to be encoded into the text
    :param num_str:
    :return: ' ' for empty or fractional numbers. For integers, formats each digit to be of the form regex form
    (OZ{k}){n} where k is the power of ten of that digit and n is the value of that digit.

    101 would return OZ O
    2356 would return OZZZ OZZZ OZZ OZZ OZ OZ OZ OZ OZ O O O O O O
    """
    if len(num_str) == 0 or not num_str.isnumeric():
        return ' '
    if num_str.startswith('.'):
        return ' '
    if num_str.startswith('-'):
        return 'NEGATIVE ' + num_to_text(num_str[1:])
    if len(num_str) == 1 and num_str == '0':
        return 'ZERO'
    return ' '.join([' '.join(int(digit) * [('O' + 'Z' * (len(num_str) - i - 1))]) for i, digit in enumerate(num_str) if digit != '0'])
def clean_string(string):
    string = html.unescape(string).replace('\n', ' ').replace('\t', ' ').lower().replace('%', ' percent ').replace('$', ' dollars ').replace("#39;", "'")\
        .replace("amp;", "&").replace("#146;", "'").replace("nbsp;", " ").replace("#36;", "$")\
        .replace("\\n", "\n").replace("quot;", "'").replace("<br />", "\n").replace('\\"', '"')\
        .replace(" @.@ ", ".").replace(" @-@ ", "-").replace(" @,@ ", ",").replace("\\", " \\ ").replace('\'s', ' ').replace('s\'', 's')\
        .replace('—', ' ').replace('﻿', ' ')
    num_split = re.split(numbers, string)
    number_indices = [(i, x) for i, x in enumerate(re.split(numbers, string)) if x is not None and re.match(numbers, x)]
    for i, num_str in number_indices:
        num_split[i] = num_to_text(num_str)
    string = ' '.join((x for x in num_split if x is not None and any((y.isalpha() for y in x)))).replace(' ', ' ').replace('-', ' ')
    string = re.sub(" {2,}", " ", re.sub(r"([/#\n])", r" \1 ", string))
    return string
def to_input_label_list(dataset):
    titles = list(map(nltk.word_tokenize, map(clean_string, (' '.join(x[0]['title']) for x in dataset))))
    labels = [int(x[1]['Label']) for x in dataset]
    return titles, labels


def prepare_data():
    db_path = os.path.join(os.path.split(os.getcwd())[0], 'Data\Headlines.db')
    real_news, [train_stock, valid_stock, test_stock] = load_data_memory_efficient(db_path, 5, (0.8, 0.1, 0.1), False, True, ["SPY", "^DJI", "NDAQ", "AAPL", "GOOG", "META"])
    title_gen = list(map(lambda x: (x[0], nltk.word_tokenize(clean_string(' '.join(x[1])))), real_news[['date_int', 'title']].reset_index().drop(columns=['index']).groupby('date_int')['title']))

    titles = []
    for idx, tokens in title_gen:
        if len(titles) < idx:
            titles.extend([None for i in range(idx - len(titles))])
        titles.append(tokens)

    word_counts = Counter(np.concatenate(list(map(lambda i: titles[i], train_stock['date_int']))))
    cutoff = 2
    # Vocabulary(word_counts, unk_cutoff=cutoff)

    # breakpoint()
    #
    # train_titles, train_labels = to_input_label_list(train)
    # validation_titles, validation_labels = to_input_label_list(validation)
    # test_titles, test_labels = to_input_label_list(test)
    # word_counts = Counter()
    # for title in train_titles:
    #     word_counts += Counter(title)
    # cuttoff = 2
    word_counts = Vocabulary(word_counts, unk_cutoff=cutoff)
    words = sorted(word_counts, key=lambda x: (x != "<UNK>", -word_counts[x], x))
    words.remove('<UNK>')
    # used as mapping of word to id
    vocab = Vocabulary({x: i for i, x in enumerate(words, start=1)}, unk_cutoff=1)
    PAD_ID = 0
    UNK_ID = vocab["<UNK>"]
    format_titles = lambda titles: map(lambda item: list(map(lambda x: vocab[x], item)), map(vocab.lookup, titles))
    title_data = {idx: np.bincount(np.concatenate(list(format_titles(titles[idx])), dtype=int), minlength=len(vocab)) for idx in range(len(titles)) if titles[idx] is not None}
    # train_data = list(zip(format_titles(train_titles), train_labels))
    # validation_data = list(zip(format_titles(validation_titles), validation_labels))
    # test_data = list(zip(format_titles(test_titles), test_labels))
    stock_format = lambda stock: np.ascontiguousarray(np.vstack([stock['date_int'].to_numpy(), stock['Label'].to_numpy()]).T.astype(int))
    return {
        'title_data': title_data,
        'train': stock_format(train_stock),
        'validation': stock_format(valid_stock),
        'test': stock_format(test_stock),
        'PAD_ID': PAD_ID,
        'UNK_ID': UNK_ID,
        'vocab': vocab
    }
if __name__ == '__main__':
    import os
    os.chdir('Notebooks')
    dat = prepare_data()