import csv
import numpy as np
import processingFunctions as pf
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn import ensemble, cluster
import random

def accuracy(y, pred):
    return np.sum(y == pred) / float(len(y))

f = open('simulated_long_data_1.csv')
reader = csv.reader(f)
headers = next(reader)
col_index = range(len(headers))
data = np.array([row for row in reader])


trajectories = pf.separate_trajectories(data)
feature_sums, TBI_class = pf.get_feature_sum_lists(trajectories)
long_study, long_study_class, short_study, short_study_class = pf.separate_by_length(TBI_class, feature_sums)
bucketed_features_long = np.array(pf.to_buckets_new(long_study, technique="new"))
TBI_0_features, TBI_1_features, TBI_2_features, TBI_3_features = pf.separate_by_class(bucketed_features_long, long_study_class)

TBI_0_features_flat, TBI_1_features_flat, TBI_2_features_flat, TBI_3_features_flat = \
    pf.flatten(pf.only_features(TBI_0_features)), pf.flatten(pf.only_features(TBI_1_features)), pf.flatten(pf.only_features(TBI_2_features)), pf.flatten(pf.only_features(TBI_3_features))

TBI_0_times_flat, TBI_1_times_flat, TBI_2_times_flat, TBI_3_times_flat = \
    pf.flatten(pf.only_times(TBI_0_features)), pf.flatten(pf.only_times(TBI_1_features)), pf.flatten(pf.only_times(TBI_2_features)), pf.flatten(pf.only_times(TBI_3_features))

POLYNOMIAL_ORDER = 3



coeff_0 = np.polyfit(TBI_0_times_flat, TBI_0_features_flat,POLYNOMIAL_ORDER)
coeff_1 = np.polyfit(TBI_1_times_flat, TBI_1_features_flat,POLYNOMIAL_ORDER)
coeff_2 = np.polyfit(TBI_2_times_flat, TBI_2_features_flat,POLYNOMIAL_ORDER)
coeff_3 = np.polyfit(TBI_3_times_flat, TBI_3_features_flat,POLYNOMIAL_ORDER)

print coeff_0
print coeff_1
print coeff_2
print coeff_3

TBI_0_pack = np.array([TBI_0_times_flat,TBI_0_features_flat]).T
TBI_1_pack = np.array([TBI_1_times_flat,TBI_1_features_flat]).T
TBI_2_pack = np.array([TBI_2_times_flat,TBI_2_features_flat]).T
TBI_3_pack = np.array([TBI_3_times_flat,TBI_3_features_flat]).T


with open('TBI_data/TBI_0.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(TBI_0_pack)

with open('TBI_data/TBI_1.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(TBI_1_pack)

with open('TBI_data/TBI_2.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(TBI_2_pack)

with open('TBI_data/TBI_3.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(TBI_3_pack)