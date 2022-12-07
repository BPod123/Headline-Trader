import pandas as pd
from tabulate import tabulate
def eval_class(labels, pred, selected_class):
    tp = ((labels == selected_class) & (pred == selected_class)).sum()
    fn = ((labels != selected_class) & (pred == selected_class)).sum()
    fp = ((labels == selected_class) & (pred != selected_class)).sum()
    recall = 0 if tp + fn == 0 else tp / (tp + fn)
    precision = 0 if tp + fp == 0 else tp / (tp + fp)
    f1 = 0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def evaluate(file_name, sub_dir):
    path = f'../Results/{sub_dir}/{file_name}'
    df = pd.read_csv(path)
    labels, predictions = df['Labels'].round(), df['Predictions'].round()
    f1_0, prec_0, rec_0 = eval_class(labels, predictions, 0)
    f1_1, prec_1, rec_1 = eval_class(labels, predictions, 1)
    f1 = [f1_0, f1_1]
    prec = [prec_0, prec_1]
    rec = [rec_0, rec_1]
    avg = lambda x: sum(x) / len(x)
    return pd.DataFrame({"Avg F1": avg(f1), "Avg Prec": [avg(prec)], "Avg Rec": [avg(rec)],
                         "Class 0 F1": [f1[0]], "Class 0 Prec": [prec[0]], "Class 0 Rec": [rec[0]],
                         "Class 1 F1": [f1[1]], "Class 1 Prec": [prec[1]], "Class 1 Rec": [rec[1]]},
                        index=[file_name[:-4]])


if __name__ == '__main__':
    print("[Average, class 0, class 1]")
    files = ['Bert', 'Bert Improved', 'BOW', 'GPT3', 'Majority_baseline', 'Random_Baseline']
    for sub_dir in ['test', 'validation', 'train']:
        summary = pd.concat([evaluate(fname + '.csv', sub_dir) for fname in files])
        print(sub_dir)
        print(tabulate(summary, tablefmt="fancy_grid", headers = summary.columns ))
        print('\n\n')