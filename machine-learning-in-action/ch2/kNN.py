from numpy import *
import operator
import matplotlib.pyplot as plt


def classify0(in_x, data_set, labels, k):
    data_set_size = data_set.shape[0]
    diff_mat = tile(in_x, (data_set_size, 1)) - data_set
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    sorted_dist_indicies = distances.argsort()
    class_count = {}
    for i in range(k):
        vote_i_label = labels[sorted_dist_indicies[i]]
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def create_data_set():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def file_to_matrix(filename):
    file = open(filename)
    array_of_lines = file.readlines()
    number_of_lines = len(array_of_lines)
    return_mat = zeros((number_of_lines, 3))
    class_label_vector = []
    index = 0
    for line in array_of_lines:
        line = line.strip()
        list_from_line = line.split('\t')
        return_mat[index, :] = list_from_line[0: 3]
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    return return_mat, class_label_vector


def auto_normalize(data_set):
    min_value = data_set.min(0)
    max_value = data_set.max(0)
    ranges = max_value - min_value
    m = data_set.shape[0]
    normal_data_set = data_set - tile(min_value, (m, 1))
    normal_data_set = normal_data_set/tile(ranges, (m, 1))
    return normal_data_set, ranges, min_value


def dating_class_test():
    ho_ratio = 0.10
    dating_data_mat, dating_labels = file_to_matrix("datingTestSet2.txt")
    norm_mat, ranges, min_value = auto_normalize(dating_data_mat)
    m = norm_mat.shape[0]
    num_test_vector = int(m * ho_ratio)
    error_count = 0.0
    for i in range(num_test_vector):
        classifier_result = classify0(norm_mat[i, :], norm_mat[num_test_vector:m, :],
                                      dating_labels[num_test_vector:m], 3)
        print("the classifier came back with: %d, the real answer is: %d"
              % (classifier_result, dating_labels[i]))
        if classifier_result != dating_labels[i]:
            error_count += 1.0
    print("the total error rate is: %f" % (error_count / float(num_test_vector)))


def draw_point():
    dating_data_mat, dating_labels = file_to_matrix("datingTestSet2.txt")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dating_data_mat[:, 0], dating_data_mat[:, 1],
               15.0 * array(dating_labels), 15.0 * array(dating_labels))
    plt.show()


def little_test_case():
    group, labels = create_data_set()
    print("The group is :")
    print(group)
    print("The labels is :")
    print(labels)
    temp = classify0([0, 0], group, labels, 3)
    print("Point[0, 0]\'s labels is :")
    print(temp)
