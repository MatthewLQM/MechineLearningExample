from numpy import *
import operator
import matplotlib.pyplot as plt
from os import listdir


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