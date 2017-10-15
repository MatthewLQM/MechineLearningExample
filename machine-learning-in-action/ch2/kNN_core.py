from numpy import *
import operator
import matplotlib.pyplot as plt
from os import listdir
import kNN_base


def classify(in_x, data_set, labels, k):
    """
    根据输入的值利用 kNN 算法进行分类，得到输入值的分类结果

    :param in_x: 输入的值
    :param data_set: 二阶矩阵，表示原始数据集（或者是说测试集）
    :param labels: 有哪些分类的结果
    :param k: kNN 算法中的参数 k（取几个最近的点）
    :return: 返回 labels 中的一个值，表示最终的结果
    """
    # 计算原始数据集的大小
    data_set_size = data_set.shape[0]
    # 计算输入值 in_x 与原始数据集的差值
    diff_mat = tile(in_x, (data_set_size, 1)) - data_set
    # 计算输入值与各个点的距离
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    # 对距离进行排序
    sorted_dist_indicies = distances.argsort()
    class_count = {}
    # 找到最近的 k 个点所属的标签是哪一个
    for i in range(k):
        vote_i_label = labels[sorted_dist_indicies[i]]
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1
    # 对所有出现过的标签进行排序
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    # 返回出现次数最多的标签
    return sorted_class_count[0][0]


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


def dating_class_test():
    ho_ratio = 0.10
    dating_data_mat, dating_labels = file_to_matrix("resources/datingTestSet2.txt")
    norm_mat, ranges, min_value = kNN_base.auto_normalize(dating_data_mat)
    m = norm_mat.shape[0]
    num_test_vector = int(m * ho_ratio)
    error_count = 0.0
    for i in range(num_test_vector):
        classifier_result = classify(norm_mat[i, :], norm_mat[num_test_vector:m, :],
                                      dating_labels[num_test_vector:m], 3)
        print("the classifier came back with: %d, the real answer is: %d"
              % (classifier_result, dating_labels[i]))
        if classifier_result != dating_labels[i]:
            error_count += 1.0
    print("the total error rate is: %f" % (error_count / float(num_test_vector)))


def draw_point():
    dating_data_mat, dating_labels = file_to_matrix("resources/datingTestSet2.txt")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dating_data_mat[:, 0], dating_data_mat[:, 1],
               15.0 * array(dating_labels), 15.0 * array(dating_labels))
    plt.show()


def little_test_case():
    """
    一个简单的测试案例
    """
    group, labels = kNN_base.create_data_set()
    print("The group is :")
    print(group)
    print("The labels is :")
    print(labels)
    temp = classify([0, 0], group, labels, 3)
    print("Point[0, 0]\'s labels is :")
    print(temp)


def img_to_vector(filename):
    return_vector = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vector[0, 32*i+j] = int(line_str[j])
    return return_vector


def handwriting_class_test():
    hw_labels = []
    training_file_list = listdir('resources/trainingDigits')
    m = len(training_file_list)
    training_mat = zeros((m, 1024))
    for i in range(m):
        file_name_str = training_file_list[i]
        file_str = file_name_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        hw_labels.append(class_num_str)
        training_mat[i, :] = img_to_vector('resources/trainingDigits/%s' % file_name_str)
    test_file_list = listdir('resources/testDigits')
    error_count = 0.0
    m_test = len(test_file_list)
    for i in range(m_test):
        file_name_str = test_file_list[i]
        file_str = file_name_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        vector_under_test = img_to_vector('resources/testDigits/%s' % file_name_str)
        classifier_result = classify(vector_under_test, training_mat, hw_labels, 3)
        print("the classifier came back with: %d, the real answer is %d"
              % (classifier_result, class_num_str))
        if classifier_result != class_num_str:
            error_count += 1.0
    print("\nthe total number of error is: %d" % error_count)
    print("\nthe total error rate is : %f" % (error_count/float(m_test)))
