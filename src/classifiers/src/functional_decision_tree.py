def read_libsvm_data(path: str) -> (list, list):
    """
    Read datset in libsvm format and turn into into a list with class value last
    :param path: of the dataset
    :return: dataset and list with the attributes names
    """
    dataset = list()
    attr_names = list()
    with open(path, "r") as f:
        for line in f.readlines():
            splitted = [x.strip() for x in line.split()[::-1]]
            parsed_line = list()
            for i in range(0, len(splitted)-1):
                parsed_line.append(splitted[i].split(":")[1])
            parsed_line.append(splitted[len(splitted)-1])
            dataset.append(parsed_line)

            if not attr_names:
                attr_names = [x.split(":")[0] for x in splitted[0:-1]]

    return dataset, attr_names


def gini_index(groups: list, class_values: list) -> float:
    """
    Calculate the Gini index for a split dataset (0 is perfect split)
    :param groups: to compute the gini index (last item is class value)
    :param class_values: class values present in the groups
    :return: gini index
    """
    gini = 0.0
    for class_value in class_values:
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            proportion = [row[-1] for row in group].count(class_value) / float(size)
            gini += (proportion * (1.0 - proportion))
    return gini


def test_split(dataset: list, index: int, value: type) -> list:
    """
    Split a dataset based on an attribute and an attribute value
    :param data: list of rows to split
    :param index: index of the attribute for splitting
    :param value: value of the attribute for splitting
    :return: two lists with each group
    """
    left = list()
    right = list()

    def _recursive_split(data):
        if len(data) > 0:
            head, *tail = data
            if head[index] < value:
                left.append(head)
            else:
                right.append(head)
            _recursive_split(tail)

    _recursive_split(dataset)

    return [left, right]


def get_split(dataset: list, class_values: list) -> dict:
    """
    Select the best split point for a dataset
    :param dataset:
    :return:  dict with best index, value, and groups
    """
    from math import inf
    best_index, best_value, best_score, best_groups = inf, inf, inf, None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(dataset, index, row[index])
            gini = gini_index(groups, class_values)
            if gini < best_score:
                best_index, best_value, best_score, best_groups = index, row[index], gini, groups
    return {'index': best_index, 'value': best_value, 'score': best_score, 'groups': best_groups}


def to_terminal(group: list, outcomes: list) -> float:
    """
    Create a terminal node value
    :param group: group of samples with target as last element
    :return: most representative class value for group
    """
    outcomes = [row[-1] for row in group]
    return [max(set(outcomes), key=outcomes.count), outcomes]


def split(node, max_depth, min_size, depth, class_values: list):
    """
    Create child splits for a node or make terminal
    :param node: current tree node to inspect
    :param max_depth: of the desired tree
    :param min_size: of each subgroup
    :param depth: current depth
    :return:
    """
    left, right = node['groups']
    del(node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right, class_values)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left, class_values), to_terminal(right, class_values)
        return

    print("Left: %s\nRight: %s\nGini: %s\n" % (left, right, gini_index([left, right], class_values)))

    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left, class_values)
    else:
        node['left'] = get_split(left, class_values)
        split(node['left'], max_depth, min_size, depth+1, class_values)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right, class_values)
    else:
        node['right'] = get_split(right, class_values)
        split(node['right'], max_depth, min_size, depth+1, class_values)


def build_tree(train: list, max_depth: int, min_size: int) -> dict:
    """
    Build a decision tree
    :param train: dataset used to build the tree
    :param max_depth: of the desired tree
    :param min_size: of the terminal nodes
    :return:
    """
    class_values = list(set(row[-1] for row in dataset))
    root = get_split(train, class_values)
    split(root, max_depth, min_size, 1, class_values)
    return root


def print_tree(node: dict, depth=0):
    """
    Print a decision tree
    :param node: root of the tree to print
    :param depth: current printing depth
    """
    if isinstance(node, dict):
        print('%s[X%d < %.3f]: Gini=%s' % (depth*' ', (node['index']+1), node['value'], node['score']))
        print_tree(node['left'], depth+1)
        print_tree(node['right'], depth+1)
    else:
        print('%s[%s]' % (depth*' ', node))


dataset = [[2.771244718, 1.784783929, 0],
    [1.728571309, 1.169761413, 0],
    [3.678319846, 2.81281357, 0],
    [3.961043357, 2.61995032, 0],
    [2.999208922, 2.209014212, 0],
    [7.497545867, 3.162953546, 1],
    [9.00220326, 3.339047188, 1],
    [7.444542326, 0.476683375, 1],
    [10.12493903, 3.234550982, 0],
    [6.642287351, 3.319983761, 1]]

# tree = build_tree(dataset, 1, 1)
# print_tree(tree)

tree = build_tree(dataset, 2, 1)
print_tree(tree)