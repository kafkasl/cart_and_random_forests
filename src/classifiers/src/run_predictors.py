from __future__ import print_function
from random_forest import RandomForest
from decision_tree import DecisionTree
from time import time
from random import shuffle
import os


def read_libsvm_data(path: str) -> (list, list):
    dataset = list()
    attr_names = list()
    class_values = set()
    with open(path, "r") as f:
        for line in f.readlines():
            splitted = [x.strip() for x in line.split()[::-1]]
            parsed_line = list()
            for i in range(0, len(splitted) - 1):
                parsed_line.append(splitted[i].split(":")[1])
            parsed_line.append(splitted[len(splitted) - 1])
            class_values.add(splitted[len(splitted) - 1])
            dataset.append(parsed_line)

            if not attr_names:
                attr_names = [x.split(":")[0] for x in splitted[0:-1]]

    return dataset, attr_names


def test_clf(classifier, root, file, number_of_folds, iterations):

        acc_train_time = 0
        acc_testing_time = 0
        data, attr_names = read_libsvm_data(os.path.join(root, file))
        acc_accuracy = 0.0
        for iteration in range(0, iterations):
            shuffle(data)

            n = len(data)
            foldSize = int(n / number_of_folds)

            for fold in range(0, number_of_folds):

                train_set = data[0:fold * foldSize-1] + data[(fold + 1) * foldSize: n]
                test_set = data[fold * foldSize: (fold + 1) * foldSize]

                b = len(train_set)
                if b > 500:
                    b = int(b/3)

                start = time()
                if classifier == 0:
                    clf = DecisionTree(train_set, attr_names, 10, 100, False)
                elif classifier == 1:
                    clf = DecisionTree(train_set, attr_names, 10, 100, True)
                elif classifier == 2:
                    clf = RandomForest(train_set, attr_names, 8, 10, 100, bagging=True, random_input=False, random_features=False)
                elif classifier == 3:
                    clf = RandomForest(train_set, attr_names, 8, 10, 100, bagging=True, random_input=False, random_features=True)
                elif classifier == 4:
                    clf = RandomForest(train_set, attr_names, 8, 10, 100, bagging=True, random_input=True, random_features=False)
                elif classifier == 5:
                    clf = RandomForest(train_set, attr_names, 8, 10, 100, bagging=True, random_input=True, random_features=True)
                training = time()

                results = list()
                for row in test_set:
                    val = clf.predict(row)
                    results.append(val == row[-1])
                testing = time()

                partial_accuracy = results.count(True)/len(results)
                acc_accuracy += partial_accuracy
                acc_train_time += (training-start)
                acc_testing_time += (testing-training)
        print("Training time: %s" % round(acc_train_time, 2))
        print("Testing time: %s" % round(acc_testing_time, 2))
        print("Total accuracy: %s" % round((acc_accuracy/(number_of_folds*iterations)), 2))

if __name__ == "__main__":
    classifiers = ["Decision Tree", "Decision Tree with random input",
                   "Tree Bagging", "Random Forest", "Forest with random input",
                   "Random Forest with random input"]

    root = "/home/hydra/miri/adm/labs/assignment.2/src/resources/"
    files = ["titanicTr.txt", "pimaTr.txt", "banknote.txt"]

    number_of_folds = 10
    iterations = 10
    clfs = [0, 1, 2, 3, 4, 5]
    for file in files:
        print("\nTesting %s\n=======================" % file)
        for classifier in clfs:
            print("\n- %s" % classifiers[classifier])
            test_clf(classifier, root=root, file=file, number_of_folds=number_of_folds, iterations=iterations)

