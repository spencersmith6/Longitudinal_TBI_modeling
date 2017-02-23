import csv
import numpy as np
import processingFunctions as pf

f = open('simulated_long_data_1.csv')
reader = csv.reader(f)
headers = next(reader)
col_index = range(len(headers))
data = np.array([row for row in reader])

# create np array of integer features
feature_headers = headers[10:]

trajectories = pf.separate_trajectories(data)
feature_sums_lists, TBI_class = pf.get_feature_sum_lists(trajectories)
long_study, long_study_class, short_study, short_study_class = pf.separate_by_length(TBI_class, feature_sums_lists)

pf.plot_trajectories(long_study, long_study_class, obs =1000 , length_of_study="long")
pf.plot_trajectories(short_study, short_study_class,obs = 1000, length_of_study="short")






