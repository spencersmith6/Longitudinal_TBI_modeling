import csv
import numpy as np
import processingFunctions as pf
import matplotlib.pyplot as plt
import scipy.stats as st


f = open('simulated_long_data_1.csv')
reader = csv.reader(f)
headers = next(reader)
col_index = range(len(headers))
data = np.array([row for row in reader])

trajectories = pf.separate_trajectories(data)
feature_sums, TBI_class = pf.get_feature_sum_lists(trajectories)
long_study, long_study_class, short_study, short_study_class = pf.separate_by_length(TBI_class, feature_sums)
bucketed_features_long = np.array(pf.to_buckets1(long_study, nbins=8))
only_features = pf.only_features(bucketed_features_long)
TBI_0_features, TBI_1_features, TBI_2_features, TBI_3_features = pf.separate_by_class(only_features, long_study_class)

######## Calculate Stats ############
TBI_0_means = np.mean(TBI_0_features, axis=0)
TBI_0_stds = np.std(TBI_0_features, axis=0)
TBI_0_count = len(TBI_0_features)

TBI_1_means = np.mean(TBI_1_features, axis=0)
TBI_1_stds = np.std(TBI_1_features, axis=0)
TBI_1_count = len(TBI_1_features)

TBI_2_means = np.mean(TBI_2_features, axis=0)
TBI_2_stds = np.std(TBI_2_features, axis=0)
TBI_2_count = len(TBI_2_features)

TBI_3_means = np.mean(TBI_3_features, axis=0)
TBI_3_stds = np.std(TBI_3_features, axis=0)
TBI_3_count = len(TBI_3_features)

num_buckets = len(TBI_0_features[1])

TBI_0 = [TBI_0_features[:,i] for i in range(num_buckets)]
TBI_1 = [TBI_1_features[:,i] for i in range(num_buckets)]
TBI_2 = [TBI_2_features[:,i] for i in range(num_buckets)]
TBI_3 = [TBI_3_features[:,i] for i in range(num_buckets)]

t_0 = st.t.ppf(1-0.025, TBI_0_count)
t_1 = st.t.ppf(1-0.025, TBI_1_count)
t_2 = st.t.ppf(1-0.025, TBI_2_count)
t_3 = st.t.ppf(1-0.025, TBI_3_count)

buckets =bucketed_features_long[0][0]

means = [TBI_0_means, TBI_1_means, TBI_2_means, TBI_3_means]
stds = [TBI_0_stds, TBI_1_stds,TBI_2_stds,TBI_3_stds]
CIs = [t_0*(TBI_0_stds/np.sqrt(TBI_0_count)) , t_1*(TBI_1_stds/np.sqrt(TBI_1_count)),t_2*(TBI_2_stds/np.sqrt(TBI_2_count)),t_3*(TBI_3_stds/np.sqrt(TBI_3_count))]

print CIs
print stds
print TBI_0_count, TBI_1_count, TBI_2_count, TBI_3_count



def plot_t_test(means, CIs):
    title = "Mean Feature Sum level with 95% CI "

    fig = plt.figure()
    plt.xlim(0, 9000)
    plt.ylim(0, 100)

    for i in range(4):
        if i == 0:
            col = 'black'
        elif i == 1:
            col = 'red'
        elif i == 2:
            col = 'green'
        else:
            col = 'blue'

        plt.errorbar(buckets, means[i], yerr=[stds[i],stds[i]], fmt='--o',color=col, label= i )

    fig.suptitle(title)
    plt.legend(loc='upper right')
    plt.ylabel('Feature Sum')
    plt.xlabel('Time After "Injury"')
    plt.show()


plot_t_test(means, CIs)





###############
# TBI_0_stats = {i:(TBI_0_means[i], TBI_0_stds[i], TBI_0_count) for i in range(num_buckets)}
# TBI_1_stats = {i:(TBI_1_means[i], TBI_1_stds[i], TBI_1_count) for i in range(num_buckets)}
# TBI_2_stats = {i:(TBI_2_means[i], TBI_2_stds[i], TBI_2_count) for i in range(num_buckets)}
# TBI_3_stats = {i:(TBI_3_means[i], TBI_3_stds[i], TBI_3_count) for i in range(num_buckets)}


# for i in range(num_buckets):
#     print "Bucket %i:" %i
#     print "TBI 0: Mean= %f ,  STD= %f , n= %i " %(TBI_0_stats[i][0], TBI_0_stats[i][1],TBI_0_stats[i][2])
#     print "TBI 1: Mean= %f ,  STD= %f , n= %i " %(TBI_1_stats[i][0], TBI_1_stats[i][1],TBI_1_stats[i][2])
#     print "TBI 2: Mean= %f ,  STD= %f , n= %i " % (TBI_2_stats[i][0], TBI_2_stats[i][1], TBI_2_stats[i][2])
#     print "TBI 3: Mean= %f ,  STD= %f , n= %i " %(TBI_3_stats[i][0], TBI_3_stats[i][1],TBI_3_stats[i][2])
#     print
#     print
###############

# TBI_0_CI = map(list, zip(*[st.t.interval(0.95, len(TBI_0[i])-1, loc=np.mean(TBI_0[i]), scale=st.sem(TBI_0[i])) if st.sem(TBI_0[i])>0 else (np.mean(TBI_0[i]),np.mean(TBI_0[i])) for i in range(num_buckets)]))
# TBI_1_CI = map(list, zip(*[st.t.interval(0.95, len(TBI_1[i])-1, loc=np.mean(TBI_1[i]), scale=st.sem(TBI_1[i])) if st.sem(TBI_1[i])>0 else (np.mean(TBI_1[i]),np.mean(TBI_1[i])) for i in range(num_buckets)]))
# TBI_2_CI = map(list, zip(*[st.t.interval(0.95, len(TBI_2[i])-1, loc=np.mean(TBI_2[i]), scale=st.sem(TBI_2[i])) if st.sem(TBI_2[i])>0 else (np.mean(TBI_2[i]),np.mean(TBI_2[i])) for i in range(num_buckets)]))
# TBI_3_CI = map(list, zip(*[st.t.interval(0.95, len(TBI_3[i])-1, loc=np.mean(TBI_3[i]), scale=st.sem(TBI_3[i])) if st.sem(TBI_3[i])>0 else (np.mean(TBI_3[i]),np.mean(TBI_3[i])) for i in range(num_buckets)]))
###########