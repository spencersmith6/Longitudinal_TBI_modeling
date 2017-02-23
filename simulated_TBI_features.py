import csv
import numpy as np
import time
from sklearn.cluster import KMeans

f = open('simulated_long_data_1.csv')
reader = csv.reader(f)
headers = next(reader)
col_index = range(len(headers))
raw_data = np.array([row for row in reader])

print headers
# create np array of integer features
feature_headers = headers[10:]
features = [[int(el.strip()) for el in row[10:len(row) - 1]] for row in raw_data]

#print features[:5]

# Create a list of (ID , epoch in hrs) tuples
id_date = [[i[0].strip(), int(time.mktime(time.strptime(i[1].strip(), '%Y-%m-%d'))) / 3600.0] for i in raw_data]
# print id_date[:5]



id_features = [id_date[i] + features[i] for i in range(len(id_date))]

# for i in id_features[:10]:
#     print i

trajectories = []  # Usable list of lists for each ID containing [ID, epoch in hrs, list of features]
id_now = id_date[0][0]
id_features_list = []
for row in range(len(id_date)):
    if id_now != id_features[row][0]:
        # add sublist to list of lists
        trajectories.append(id_features_list)

        # reinitiallize
        id_now = id_features[row][0]
        id_features_list = []

    id_features_list.append(id_features[row])

trajectories = np.array(trajectories)
# for i in trajectories[:5]:
#     print i

print len(trajectories)


def get_diff(trajectory):
    length = len(trajectory)

    avg_feature_diff_list = []
    last_feature = np.array(trajectory[0][2:])
    last_time = trajectory[0][1]

    for row in range(length):
        if row != 0:
            time_diff = trajectory[row][1] - last_time
            avg_feature_diff = (np.array(trajectory[row][2:]) - last_feature) / time_diff
            avg_feature_diff_list.append(list(trajectory[row][:2]) +list(avg_feature_diff))

            # reinitialization
            last_feature = np.array(trajectory[row][2:])
            last_time = trajectory[row][1]

    return avg_feature_diff_list

# result =get_diff(trajectories[0])
# feature_results = np.array([i[2:] for i in result])
# mean_diff = np.mean(feature_results, 0)
# print mean_diff
#
# print
# result2 = get_diff(result)
# feature_results2 = np.array([i[2:] for i in result2])
# mean_diff2 = np.mean(feature_results2, 0)
# print mean_diff2

ids = []
features = []
for trajectory in trajectories:
    result = get_diff(trajectory)
    ids.append(trajectory[0][0])
    feature_results = np.array([i[2:] for i in result])
    mean_diff = np.mean(feature_results, 0)
    features.append(mean_diff)

features = np.array(features)

print len(ids), len(features)

kmeans = KMeans(n_clusters=3)
kmeans.fit(features)
labels =kmeans.labels_

count= 0
for i in range(len(labels)):
    if labels[i] == int(raw_data[i][8]):
        count += 1
    print labels[i], ids[i], raw_data[i][8]

print count/float(len(labels))