import csv
import numpy as np
import processingFunctions as pf
import scipy.stats as st
import matplotlib.pyplot as plt



f = open('simulated_long_data_1.csv')
reader = csv.reader(f)
headers = next(reader)
col_index = range(len(headers))
data = np.array([row for row in reader])

trajectories = pf.separate_trajectories(data)
feature_sums, TBI_class = pf.get_feature_sum_lists(trajectories)
long_study, long_study_class, short_study, short_study_class = pf.separate_by_length(TBI_class, feature_sums)
bucketed_features_long = np.array(pf.to_buckets1(long_study, nbins=8))

only_features = np.array((bucketed_features_long[:, 1]))

long_study_class = np.array(long_study_class)
TBI_0_features = only_features[long_study_class == 0]
TBI_1_features = only_features[long_study_class == 1]
TBI_2_features = only_features[long_study_class == 2]
TBI_3_features = only_features[long_study_class == 3]

num_buckets = len(TBI_0_features[1])


TBI_0 = [TBI_0_features[:,i] for i in range(num_buckets)]
TBI_1 = [TBI_1_features[:,i] for i in range(num_buckets)]
TBI_2 = [TBI_2_features[:,i] for i in range(num_buckets)]
TBI_3 = [TBI_3_features[:,i] for i in range(num_buckets)]


TBI_0_CI = [st.t.interval(0.95, len(TBI_0[i])-1, loc=np.mean(TBI_0[i]), scale=st.sem(TBI_0[i])) if st.sem(TBI_0[i])>0 else (np.mean(TBI_0[i]),np.mean(TBI_0[i])) for i in range(num_buckets)]
TBI_1_CI = [st.t.interval(0.95, len(TBI_1[i])-1, loc=np.mean(TBI_1[i]), scale=st.sem(TBI_1[i])) if st.sem(TBI_1[i])>0 else (np.mean(TBI_1[i]),np.mean(TBI_1[i])) for i in range(num_buckets)]
TBI_2_CI = [st.t.interval(0.95, len(TBI_2[i])-1, loc=np.mean(TBI_2[i]), scale=st.sem(TBI_2[i])) if st.sem(TBI_2[i])>0 else (np.mean(TBI_2[i]),np.mean(TBI_2[i])) for i in range(num_buckets)]
TBI_3_CI = [st.t.interval(0.95, len(TBI_3[i])-1, loc=np.mean(TBI_3[i]), scale=st.sem(TBI_3[i])) if st.sem(TBI_3[i])>0 else (np.mean(TBI_3[i]),np.mean(TBI_3[i])) for i in range(num_buckets)]



print map(list, zip(*TBI_0_CI))



#
# for i in range(num_buckets):
#     if st.sem(TBI_2[i])>0:
#         print "Bucket %i: %s" %(i, str(st.t.interval(0.95, len(TBI_2[i])-1, loc=np.mean(TBI_2[i]), scale=st.sem(TBI_2[i]))))
#     else:
#         print "Bucket %i: %s" %(i, (np.mean(TBI_2[i]), np.mean(TBI_2[i])))
#
