from numpy import *
import operator


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
