import kNN
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

group, labels = kNN.create_data_set()
print("The group is :")
print(group)
print("The labels is :")
print(labels)
temp = kNN.classify0([0, 0], group, labels, 3)
print("Point[0, 0]\'s labels is :")
print(temp)

ho_ratio = 0.10
dating_data_mat, dating_labels = kNN.file_to_matrix("datingTestSet2.txt")
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dating_data_mat[:, 0], dating_data_mat[:, 1],
           15.0*array(dating_labels), 15.0*array(dating_labels))
#plt.show()
norm_mat, ranges, min_value = kNN.auto_normalize(dating_data_mat)

m = norm_mat.shape[0]
num_test_vector = int(m*ho_ratio)
error_count = 0.0
for i in range(num_test_vector):
    classifier_result = kNN.classify0(norm_mat[i, :], norm_mat[num_test_vector:m, :],
                                      dating_labels[num_test_vector:m], 3)
    print("the classifier came back with: %d, the real answer is: %d"
          % (classifier_result, dating_labels[i]))
    if classifier_result != dating_labels[i]:
        error_count += 1.0
print("the total error rate is: %f" % (error_count/float(num_test_vector)))
