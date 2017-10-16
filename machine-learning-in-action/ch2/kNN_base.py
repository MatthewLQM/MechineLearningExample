from numpy import *
import matplotlib.pyplot as plt


def create_data_set():
    """
    创建一个简单的测试数据，group 代表原始数据集，labels 表示原始数据的分类

    :return: 数据集以及标签
    """
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def auto_normalize(data_set):
    """
    把一个测试数据集归一化
    归一化的方法：new = (old - min)/(max - min)

    :param data_set:测试数据集
    :return normal_date_set 测试数据集归一化后的结果
    :return ranges 测试数据集的区间大小
    :return min_value 测试数据集中的最小值
    """
    min_value = data_set.min(0)
    max_value = data_set.max(0)
    ranges = max_value - min_value
    m = data_set.shape[0]
    normal_data_set = data_set - tile(min_value, (m, 1))
    normal_data_set = normal_data_set / tile(ranges, (m, 1))
    return normal_data_set, ranges, min_value


def file_to_matrix(filename):
    """
    根据文件名，把文件的内容转换成我们需要的矩阵
    :param filename: 文件名
    :return return_mat: 文件中存在的矩阵
    :return class_label_vector: 所有的 label
    """
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


def draw_point():
    """
    根据信息画出图像
    """
    dating_data_mat, dating_labels = file_to_matrix("resources/datingTestSet2.txt")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dating_data_mat[:, 0], dating_data_mat[:, 1],
               15.0 * array(dating_labels), 15.0 * array(dating_labels))
    plt.show()
