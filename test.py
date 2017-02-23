
import csv
import numpy as np
import processingFunctions as pf


f = open('simulated_long_data_1.csv')
reader = csv.reader(f)
headers = next(reader)
col_index = range(len(headers))
data = np.array([row for row in reader])

trajectories = pf.separate_trajectories(data)
feature_sums, TBI_class = pf.get_feature_sum_lists(trajectories)
long_study, long_study_class, short_study, short_study_class = pf.separate_by_length(TBI_class, feature_sums)
bucketed_features_long = np.array(pf.to_buckets_new(long_study, technique="new"))


features_flat = [i for sublist in bucketed_features_long  for i in sublist[1]]
times_flat = [i for sublist in bucketed_features_long  for i in sublist[0]]

print np.polyfit(times_flat, features_flat, 3)
