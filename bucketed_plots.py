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

bucketed_features_long, compression_long = pf.to_buckets_new(long_study, nbins=17, evaluate=True, technique="new")
pf.plot_trajectories(bucketed_features_long, long_study_class, obs =1000 , length_of_study="long", comments="Bucketed_new")

# bucketed_features_short, compression_short = pf.to_buckets1(short_study, nbins=4, evaluate=True)
# pf.plot_trajectories(bucketed_features_short, short_study_class, obs =1000 , length_of_study="short", comments="3_Buckets")



# print "Average Compression per Bucket ('Short' Study): "
# print "\t" + str(compression_short)
print
print "Average Compression per Bucket ('Long' Study): "
print "\t" + str(compression_long)