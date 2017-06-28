from decision_tree import DecisionTree
from random import randint
from multiprocessing import Process, Queue, cpu_count


class RandomForest(object):

    def __init__(self, data, attr_names, b: int, max_depth=1, min_size=1, bagging=False, random_input=False, random_features=False):
        """

        :param data: dataset for training
        :param attr_names: names of the dataset (for string representation purposes only)
        :param b: number of trees to train
        :param max_depth: max depth of each trained tree
        :param min_size: minimum elements in a terminal node of the trained trees
        :param bagging: perform random sampling with replacement for each trained tree
        :param random_input: instead of getting best split, select randomly one among the best 20
        :param random_features: instead of evaluating all features in each split, select randomly a subset of features
        """
        self.data = data
        self.attr_names = attr_names
        self.class_values = set([row[-1] for row in data])
        self.tree = list()

        cpus = 2*cpu_count()
        batches = int(b / cpus)

        # Divide the training into batches in order to avoid creating to many processes for parallel training
        for batch in range(0, batches):
            queues = []
            processes = []
            # Train simultaneously as many trees as CPUs has the computer
            for i in range(0, cpus):
                q = Queue()
                p = Process(target=RandomForest.train_tree, args=(q, data, attr_names, b, max_depth, min_size,
                                                                  bagging, random_input, random_features))
                p.start()
                queues.append(q)
                processes.append(p)

            for i in range(0, cpus):
                self.tree.append(queues[i].get())
                processes[i].join()
                # train_set = RandomForest.sample_with_replacement(data, b)
                # self.tree.append(DecisionTree(train_set, max_depth, min_size, random_input, random_features))

    @staticmethod
    def train_tree(q, train_set, attr_names, b, max_depth, min_size, bagging, random_input, random_features):
        """
        Auxiliar method coded to train each Decision Tree in parallel
        :param q: queue used to send the trained tree to the master
        :param train_set: training set
        :param attr_names: same as in RandomForest constructor
        :param b: same as in RandomForest constructor
        :param max_depth: same as in RandomForest constructor
        :param min_size: same as in RandomForest constructor
        :param bagging: same as in RandomForest constructor
        :param random_input: same as in RandomForest constructor
        :param random_features: same as in RandomForest constructor
        """
        if bagging:
            train_set = RandomForest.sample_with_replacement(train_set, b)
        dt = DecisionTree(train_set, attr_names, max_depth, min_size, random_input, random_features)
        q.put(dt)

    @staticmethod
    def sample_with_replacement(data: list, b: int):
        """
        Return a sample of the dataset for bagging. Uses random sampling with replacement
        :param data: data to draw elements from
        :param b: number of elements to draw
        :return: sampled data
        """
        sample = list()
        # print("Len %s, b %s" % (len(data), b))
        b = len(data) - 1
        for i in range(0, b):
            index = randint(0, b)
            sample.append(data[index])
        return sample

    def predict(self, row):
        """
        Method to compute the prediction of a forest by returning the class with more occurrences
        :param row: element to classify
        :return: predicted class
        """
        results = list()
        for i in range(0, len(self.tree)):
            results.append(self.tree[i].predict(row))

        return max(results)
