import kNN_core
import kNN_base
from numpy import *
from os import listdir


def date_classifier():
    """
    手动输入测试用例，输出测试结果
    """
    result_list = ['not at all', 'in small doses', 'in large doses']
    percent_tats = float(input("percentage of time spent playing video games?"))
    fly_miles = float(input("frequent flier miles earned per year?"))
    ice_cream = float(input("liters of ice cream consumed per year?"))
    dating_data_mat, dating_labels = kNN_base.file_to_matrix("datingTestSet2.txt")
    normal_mat, ranges, min_value = kNN_base.auto_normalize(dating_data_mat)
    in_array = array([fly_miles, percent_tats, ice_cream])
    classifier_result = kNN_core.classify((in_array - min_value) / ranges, normal_mat, dating_labels, 3)
    print("You will probably like this person:", result_list[classifier_result - 1])


def little_test_case():
    """
    一个简单的测试案例
    """
    group, labels = kNN_base.create_data_set()
    print("The group is :")
    print(group)
    print("The labels is :")
    print(labels)
    temp = kNN_core.classify([0, 0], group, labels, 3)
    print("Point[0, 0]\'s labels is :")
    print(temp)


def img_to_vector(filename):
    """
    把一个文件中的信息转换成向量
    :param filename: 文件名
    :return: 得到的向量
    """
    return_vector = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vector[0, 32*i+j] = int(line_str[j])
    return return_vector


def handwriting_class_test():
    """
    识别手写数字的测试代码
    """
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
        classifier_result = kNN_core.classify(vector_under_test, training_mat, hw_labels, 3)
        print("the classifier came back with: %d, the real answer is %d"
              % (classifier_result, class_num_str))
        if classifier_result != class_num_str:
            error_count += 1.0
    print("\nthe total number of error is: %d" % error_count)
    print("\nthe total error rate is : %f" % (error_count/float(m_test)))


def dating_class_test():
    """
    测试函数
    """
    ho_ratio = 0.10
    dating_data_mat, dating_labels = kNN_base.file_to_matrix("resources/datingTestSet2.txt")
    norm_mat, ranges, min_value = kNN_base.auto_normalize(dating_data_mat)
    m = norm_mat.shape[0]
    num_test_vector = int(m * ho_ratio)
    error_count = 0.0
    for i in range(num_test_vector):
        classifier_result = kNN_core.classify(norm_mat[i, :], norm_mat[num_test_vector:m, :],
                                              dating_labels[num_test_vector:m], 3)
        print("the classifier came back with: %d, the real answer is: %d"
              % (classifier_result, dating_labels[i]))
        if classifier_result != dating_labels[i]:
            error_count += 1.0
    print("the total error rate is: %f" % (error_count / float(num_test_vector)))


handwriting_class_test()
date_classifier()
