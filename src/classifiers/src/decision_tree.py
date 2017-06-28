from math import inf, floor, sqrt
from operator import itemgetter
from random import randint, sample


class DecisionTree(object):

    def __init__(self, data, attr_names, max_depth=1, min_size=1, random_input=False, random_features=False):
        self.data = data
        self.attr_names = attr_names
        self.class_values = set([row[-1] for row in data])
        self.random_input = random_input
        self.min_size = min_size
        self.max_depth = max_depth

        self.random_features = random_features

        # Tree construction
        self.root = self.build_tree(data)


    @staticmethod
    def read_libsvm_data(path: str) -> (list, list, list):
        """
        Read a dataset in libsvm format and convert it into the DecisionTree input format
        :param path: of the file
        :return: the read dataset in DecisionTree accepted format
        """
        dataset = list()
        attr_names = list()
        class_values = set()
        with open(path, "r") as f:
            for line in f.readlines():
                splitted = [x.strip() for x in line.split()[::-1]]
                parsed_line = list()
                for i in range(0, len(splitted)-1):
                    parsed_line.append(splitted[i].split(":")[1])
                parsed_line.append(splitted[len(splitted)-1])
                class_values.add(splitted[len(splitted)-1])
                dataset.append(parsed_line)

                if not attr_names:
                    attr_names = [x.split(":")[0] for x in splitted[0:-1]]

        return dataset

    def information_of_split(self, left: list, right: list) -> float:

        """
        Compute the information entropy of the left-right split
        :param left: left group
        :param right: right group
        :return: entropy information of the split
        """
        total = left + right
        g = self.gini_index(total)
        q = len(left) / len(total)
        gl = self.gini_index(left)
        gr = self.gini_index(right)

        return g - (q*gl) - (1-q)*gr

    def gini_index(self, node: list) -> float:
        """
        Calculate the Gini index for a node (0 is perfect split)
        :param node: upon which to compute the Gini index
        :return: gini index
        """
        gini = 0.0
        if len(node) > 0:
            pn = [row[-1] for row in node].count(node[0][-1]) / float(len(node))
            gini = 1 - pn**2 - (1-pn)**2
        return gini

    def test_split(self, dataset: list, index: int, value: type) -> list:
        """
        Split a dataset based on an attribute and an attribute value
        :param data: list of rows to split
        :param index: index of the attribute for splitting
        :param value: value of the attribute for splitting
        :return: two lists with each group
        """
        left = list()
        right = list()

        for element in dataset:

            if element[index] < value:
                left.append(element)
            else:
                right.append(element)

        return [left, right]

    def get_split(self, dataset: list) -> dict:
        """
        Select the split point for a dataset (can use best split or best out of 20 when using randomized input)
        :param dataset:
        :return:  dict with best index, value, and groups
        """
        if len(dataset) <= self.min_size:
            return TerminalNode(dataset)

        def get_best_split() -> tuple:
            """
            Select the best split point for a dataset (no randomized input tree)
            :param dataset:
            :return:  dict with best index, value, and groups
            """
            best_index, best_value, best_score, best_groups = inf, inf, 0, None
            pairs_tried = set()
            indices = get_indices()
            for index in indices:
                for row in dataset:
                    if not (index, row[index]) in pairs_tried:
                        left, right = self.test_split(dataset, index, row[index])
                        info = self.information_of_split(left, right)
                        if info > best_score:
                            best_index, best_value, best_score, best_groups = index, row[index], info, [left, right]
                        pairs_tried.add((index, row[index]))
            if best_groups:
                return best_index, best_value, best_score, best_groups
            else:
                return None, None, None, None

        def get_rand_of_best_20_split() -> tuple:
            """
            Select the best split point for a dataset
            :param dataset:
            :return:  dict with best index, value, and groups
            """
            candidates = list()
            pairs_tried = set()
            indices = get_indices()
            for index in indices:
                for row in dataset:
                    if not (index, row[index]) in pairs_tried:
                        left, right = self.test_split(dataset, index, row[index])
                        info = self.information_of_split(left, right)
                        if info > 0:
                            candidates.append((index, row[index], info, [left, right]))
                        pairs_tried.add((index, row[index]))
            if candidates:
                candidates.sort(key=itemgetter(2))
                limit = min(len(candidates), 20)
                chosen = randint(0, limit)

                return candidates[-chosen]
            else:
                return None, None, None, None

        def get_indices() -> tuple:
            """
            Return the indices of all attributes or a random subset of it
            :param dataset:
            :return:  dict with best index, value, and groups
            """

            indices = range(len(dataset[0]) - 1)
            if self.random_features: # we are using randomly subseting the features to be evaluated for split
                return sample(indices, floor(sqrt(len(indices)))) # we use sqrt(features)
            else: # return all features for split evaluation
                return indices

        if not self.random_input:
            best_index, best_value, best_score, best_groups = get_best_split()
        else:
            best_index, best_value, best_score, best_groups = get_rand_of_best_20_split()

        if not best_groups:
            return TerminalNode(dataset)
        else:
            return SplitNode(index=best_index, value=best_value, score=best_score, left=best_groups[0],
                             right=best_groups[1], name=self.attr_names[best_index])

    def split(self, node, depth):
        """
        Create child splits for a node or make terminal
        :param node: current tree node to inspect
        :param max_depth: of the desired tree
        :param min_size: of each subgroup
        :param depth: current depth
        :return:
        """
        if isinstance(node, SplitNode):
            left, right = node.left, node.right

            # check for a no split
            if not left or not right:
                node.left = node.right = TerminalNode(left + right)
                return
            # check for max depth
            if depth >= self.max_depth:
                node.left, node.right = TerminalNode(left), TerminalNode(right)
                return

            # process left child
            node.left = self.get_split(left)
            self.split(node.left, depth+1)

            # process right child
            node.right = self.get_split(right)
            self.split(node.right, depth+1)



    def build_tree(self, train: list) -> dict:
        """
        Build a decision tree
        :param train: dataset used to build the tree
        :param max_depth: of the desired tree
        :param min_size: of the terminal nodes
        :return:
        """
        root = self.get_split(train)
        self.split(root, 1)
        return root

    @classmethod
    def print(cls, node, depth=0):
        """
        Print a decision tree
        :param node: root of the tree to print
        :param depth: current printing depth
        """
        if isinstance(node, DecisionTree):
            cls.print(node.root, 0)
        elif isinstance(node, SplitNode):
            cls.print('%s%s' % (depth * ' ', node))
            cls.print(node.left, depth + 1)
            cls.print(node.right, depth + 1)
        else:
            print('%s%s' % (depth * ' ', node))

    def predict(self, sample):
        def _predict(node, row):
            """
            Make a prediction with a decision tree
            :param row:
            :return:
            """
            if isinstance(node, TerminalNode):
                return node.representative()
            if row[node.index] < node.value:
                if isinstance(node.left, SplitNode):
                    return _predict(node.left, row)
                else:
                    return node.left.representative()
            else:
                if isinstance(node.right, SplitNode):
                    return _predict(node.right, row)
                else:
                    return node.right.representative()

        return _predict(self.root, sample)

    def __eq__(self, other):
        self.data = other.data
        self.attr_names = other.attr_names
        self.class_values = other.class_values
        self.root = other.root


# Class representing a leaf node with no further splits
class TerminalNode(object):
    def __init__(self, group: list):
        """
        Terminal node constructor
        :param group: of elements of the node
        :param representative: is set to none in order to compute the representative in the __str__ function (when we
        want to print) instead of during the tree construction
        """
        self.group = group
        self._representative = None

    def representative(self):
        """
        Get the most representative element of the set (the class that has more occurrences among the group)
        :return: representative element class
        """
        if not self._representative: # if the value has not already been computed, do it an cache it
            outcomes = [row[-1] for row in self.group]
            self._representative = max(set(outcomes), key=outcomes.count)
        return self._representative

    def __str__(self):
        attrs = []
        for attr in range(0, len(self.group[0])-1):
            attrs.append(max(set([row[attr] for row in self.group])))
        outcomes = [row[-1] for row in self.group]
        return "[%s] (%s): %s" % (self.representative(), round(outcomes.count(self.representative())/len(self.group),2), attrs)

    def __eq__(self, other):
        """
        Definition of equal operator for TerminalNode, a node is equal to another if their sets are equal
        :param other:
        """
        self.group = other.group


# Class representing an split of the Decision Tree
class SplitNode(object):
    def __init__(self, left: list, right: list, index: int, value, score: float, name: str):
        """

        :param left: left group of the split
        :param right: right group of the split
        :param index: of the parameter upon which the split was done
        :param value: of the parameter upon which the split was done
        :param score: information entropy score of the split
        :param name: of the parameter upon which the split was done
        """
        self.index = index
        self.left = left
        self.right = right
        self.value = value
        self.score = score
        self.name = name

    def __str__(self):
        return '[Attr:%s - Val:%s]: Score=%s' % (self.name, self.value, round(self.score, 3))

    def __eq__(self, other):
        """
        Definition of equal operator for SplitNode, a node is equal to another if all their parameters are equal
        :param other:
        """
        self.index = other.index
        self.left = other.left
        self.right = other.right
        self.value = other.value
        self.score = other.score