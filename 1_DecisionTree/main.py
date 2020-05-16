import numpy as np
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
From decisionTree import Node, DecisionTree

# Help function
def evaluate_tree(clf, kfold):
    results = []
    train_time = []
    test_time = []
    for train_idx, test_idx in kf.split(data.data):
        X_train, y_train = data.data[train_idx], data.target[train_idx]
        X_test, y_test = data.data[test_idx], data.target[test_idx]
        t1 = time.time()
        clf.fit(X_train, y_train)
        t2 = time.time()
        train_time.append(t2 - t1)
        score = clf.score(X_test, y_test)
        test_time.append(time.time() - t2)
        results.append(score)
    return results, train_time, test_time

def avg_performance(x):
    assert isinstance(x, list)
    return np.average(np.array(x))

if __name__ == '__main__':
    data = load_breast_cancer()
    kf = KFold(n_splits=5, random_state=2020, shuffle=True)
    # Sklearn built-in DecisionTreeClassifier
    clf1 = DecisionTreeClassifier(random_state=508)
    builtin_results, builtin_train_time, builtin_test_time = evaluate_tree(clf1, kf)
    # My DecisionTreeClassifier
    clf2 = DecisionTree('id3')
    my_results, my_train_time, my_test_time = evaluate_tree(clf2, kf)
    print(builtin_results, my_results)
