import kNN
from numpy import *

result_list = ['not at all', 'in small doses', 'in large doses']
percent_tats = float(input("percentage of time spent playing video games?"))
fly_miles = float(input("frequent flier miles earned per year?"))
ice_cream = float(input("liters of ice cream consumed per year?"))
dating_data_mat, dating_labels = kNN.file_to_matrix("datingTestSet2.txt")
normal_mat, ranges, min_value = kNN.auto_normalize(dating_data_mat)
in_array = array([fly_miles, percent_tats, ice_cream])
classifier_result = kNN.classify0((in_array - min_value)/ranges, normal_mat, dating_labels, 3)
print("You will probably like this person: ", result_list[classifier_result - 1])


