import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from decision_tree import *

from decision_tree import DecisionTree


class DecisionTreeTests(unittest.TestCase):

    def test_read_data(self):
        result_data = [['FALSE', 'high', 'hot', 'sunny', 'no'],
                 ['TRUE', 'high', 'hot', 'sunny', 'no'],
                 ['FALSE', 'high', 'hot', 'overcast', 'yes'],
                 ['FALSE', 'high', 'mild', 'rainy', 'yes'],
                 ['FALSE', 'normal', 'cool', 'rainy', 'yes'],
                 ['TRUE', 'normal', 'cool', 'rainy', 'no'],
                 ['TRUE', 'normal', 'cool', 'overcast', 'yes'],
                 ['FALSE', 'high', 'mild', 'sunny', 'no'],
                 ['FALSE', 'normal', 'cool', 'sunny', 'yes'],
                 ['FALSE', 'normal', 'mild', 'rainy', 'yes'],
                 ['TRUE', 'normal', 'mild', 'sunny', 'yes'],
                 ['TRUE', 'high', 'mild', 'overcast', 'yes'],
                 ['FALSE', 'normal', 'hot', 'overcast', 'yes'],
                 ['TRUE', 'high', 'mild', 'rainy', 'no']]
        self.assertEquals(DecisionTree.read_libsvm_data("../resources/weatherNominalTr.txt"), result_data)

    def test_gini_index_1(self):
        data = [[[1, 1], [1, 1]],  [[0, 0], [0, 0]]]
        self.failUnless(DecisionTree([item for sublist in data for item in sublist] ).gini_index(data) == 0.0)

    def test_gini_index_2(self):
        data = [[[1, 1], [0, 0]], [[1, 1], [0, 0]]]
        self.failUnless(DecisionTree([item for sublist in data for item in sublist] ).gini_index(data) == 1.0)

    def test_gini_index_3(self):
        data = [[[1, 0], [1, 0]], [[1, 1], [0.3, 1], [0, 0], [0.6, 1]]]
        self.failUnless(DecisionTree([item for sublist in data for item in sublist]).gini_index(data) == 0.375)

    def test_gini_index_4(self):
        data = [[[1, 0], [1, 0], [1, 0], [0.3, 1]], [[0, 0], [0.6, 1]]]
        result = DecisionTree([item for sublist in data for item in sublist]).gini_index(data)
        print(result)
        self.failUnless(result == 0.875)

    def test_split_1(self):
        to_split = [[1, 0], [1, 0], [1, 1], [0.3, 1], [0, 0], [0.6, 1]]
        splitted = [[[0.3, 1], [0, 0]], [[1, 0], [1, 0], [1, 1], [0.6, 1]]]
        self.assertEquals(DecisionTree(to_split).test_split(to_split, 0, 0.5), splitted)

    # def test_split_2(self):
    #     to_split = [[1], [2], [3]]
    #     with self.assertRaises(IndexError) as context:
    #         DecisionTree(to_split).test_split(to_split, 1, 0.5)

    # def test_get_split(self):
    #     dataset = [[2.771244718, 1.784783929, 0],
    #                [1.728571309, 1.169761413, 0],
    #                [3.678319846, 2.81281357, 0],
    #                [3.961043357, 2.61995032, 0],
    #                [2.999208922, 2.209014212, 0],
    #                [7.497545867, 3.162953546, 1],
    #                [9.00220326, 3.339047188, 1],
    #                [7.444542326, 0.476683375, 1],
    #                [10.12493903, 3.234550982, 1],
    #                [6.642287351, 3.319983761, 1]]
    #
    #     split = DecisionTree(dataset).get_split(dataset)
    #     group_1 = [[2.771244718, 1.784783929, 0], [1.728571309, 1.169761413, 0], [3.678319846, 2.81281357, 0],
    #                [3.961043357, 2.61995032, 0], [2.999208922, 2.209014212, 0]]
    #     group_2 = [[7.497545867, 3.162953546, 1], [9.00220326, 3.339047188, 1], [7.444542326, 0.476683375, 1],
    #                [10.12493903, 3.234550982, 1], [6.642287351, 3.319983761, 1]]
    #     result = [0, 6.642, group_1, group_2]
    #     self.assertEquals([split['index'], round(split['value'], 3), split['groups'][0], split['groups'][1]], result)


    def test_to_terminal_1(self):
        dataset = [[2.771244718, 1.784783929, 0],
                   [1.728571309, 1.169761413, 0],
                   [3.678319846, 2.81281357, 0],
                   [3.961043357, 2.61995032, 0],
                   [2.999208922, 2.209014212, 1],
                   [7.497545867, 3.162953546, 1],
                   [9.00220326, 3.339047188, 1],
                   [7.444542326, 0.476683375, 1],
                   [10.12493903, 3.234550982, 1],
                   [6.642287351, 3.319983761, 1]]

        self.assertEquals(TerminalNode(dataset).group, dataset)


    # def test_build_tree(self):
    #     n0 = TerminalNode([0])
    #     n1 = TerminalNode([0, 0, 0, 0])
    #     n2 = TerminalNode([1])
    #     n3 = TerminalNode([1, 1, 1, 1])
    #     sn0 = SplitNode[0]
    #     dataset = [[2.771244718, 1.784783929, 0],
    #                [1.728571309, 1.169761413, 0],
    #                [3.678319846, 2.81281357, 0],
    #                [3.961043357, 2.61995032, 0],
    #                [2.999208922, 2.209014212, 0],
    #                [7.497545867, 3.162953546, 1],
    #                [9.00220326, 3.339047188, 1],
    #                [7.444542326, 0.476683375, 1],
    #                [10.12493903, 3.234550982, 0],
    #                [6.642287351, 3.319983761, 1]]
    #
    #     tree = DecisionTree(dataset, 2, 1)
    #     tree.print(tree)


# print('Split: [X%d < %.3f]' % ((split['index']+1), split['value']))
def main():
    unittest.main()

if __name__ == '__main__':
    main()