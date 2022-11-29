from data import load_data_from_pickle
import random
def get_majority_class(ds):
    gt = list()
    for new, outcome in ds:
        outcome = 1 if outcome.Label else -1
        gt.append(outcome)
    return 1 if sum(gt) > 0 else -1

def majority_class_eval(ds, class_label):
    pred = list()
    gt = list()
    correct = 0
    for news, outcome in ds:
        outcome = 1 if outcome.Label else -1
        if outcome == class_label:
            correct += 1
        pred.append(class_label)
        gt.append(outcome)
    # print("true num up labels", len([1 for i in gt if i == 1]))
    # print("predicted up labels", len([1 for i in pred if i == 1]))
    # print("num samples", len(gt))
    # print(correct / len(gt))
    return correct / len(gt)
        

def random_eval(ds, seed=0):
    """ get the accuracy of random eval """
    random.seed(seed)
    pred = list()
    gt = list()
    correct = 0
    for news, outcome in ds:
        outcome = 1 if outcome.Label else -1
        p = random.choice([-1, 1])
        if outcome == p:
            correct += 1
        pred.append(p)
        gt.append(outcome)
    # print("true num up labels", len([1 for i in gt if i == 1]))
    # print("predicted up labels", len([1 for i in pred if i == 1]))
    # print("num samples", len(gt))
    # print(correct / len(gt))
    return correct / len(gt)
def random_eval_stats(ds):
    """ get the mean and std accuracy of random eval for 100 random seeds """
    accuracies = list()
    for i in range(100):
        accuracies.append(random_eval(ds, seed=i))
    mean_acc = sum(accuracies) / len(accuracies)
    # print("mean accuracy", mean_acc)
    std_acc = (sum([(i - mean_acc)**2 for i in accuracies]) / len(accuracies))**0.5
    # print("std accuracy", std_acc)
    return mean_acc, std_acc
if __name__ == "__main__":
    ds = load_data_from_pickle("Headlines.db")
    train_ds = ds[0]
    val_ds = ds[1]
    test_ds = ds[2]
    print("majority class for ds's")
    print(get_majority_class(train_ds))
    print(get_majority_class(val_ds))
    print(get_majority_class(test_ds))
    print("evaluate on majority class for ds's")
    print(majority_class_eval(train_ds, 1))
    print(majority_class_eval(val_ds, get_majority_class(train_ds)))
    print(majority_class_eval(test_ds, get_majority_class(val_ds)))
    print(majority_class_eval(test_ds, get_majority_class(train_ds)))
    print("random eval")
    print(random_eval(train_ds))
    print(random_eval(val_ds))
    print(random_eval(test_ds))
    print("random eval stats")
    print(random_eval_stats(train_ds))
    print(random_eval_stats(val_ds))
    print(random_eval_stats(test_ds))
    

