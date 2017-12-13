from math import log


def calculate_shannon_entropy(data_set):
    num_entries = len(data_set)
    label_counts = {}
    for feature_vector in data_set:
        current_label = feature_vector[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_entropy = 0.0
    for key in label_counts:
        prob = float(label_counts[key])/num_entries
        shannon_entropy -= prob * log(prob, 2)
    return shannon_entropy


def create_data_set():
    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return data_set, labels


def split_data_set(data_set, axis, value):
    return_data_set = []
    for feature_vec in data_set:
        if feature_vec[axis] == value:
            reduced_feature_vector = feature_vec[:axis]
            reduced_feature_vector.extend(feature_vec[axis+1:])
            return_data_set.append(reduced_feature_vector)
    return return_data_set

def chooes_best_feature_to_split(data_set):
    num_features = len(data_set[0]) - 1
    base_entropy = calculate_shannon_entropy(data_set)
    best_info_gain = 0.0
    best_feature = -1
    for i in range(num_features):
        feature_list = [example[i] for example in data_set]
        unique_values = set(feature_list)
        new_entropy = 0.0
        for value in unique_values:
            sub_data_set = split_data_set(data_set, i, value)
            prob = len(sub_data_set)/float(len(data_set))
            new_entropy += prob * calculate_shannon_entropy(sub_data_set)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


my_data, my_labels = create_data_set()
print(my_data)
print(calculate_shannon_entropy(my_data))
my_data[0][-1] = 'maybe'
print(my_data)
print(calculate_shannon_entropy(my_data))

