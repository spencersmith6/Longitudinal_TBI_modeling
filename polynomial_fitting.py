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

POLYNOMIAL_ORDER = 2

coeff_all = [np.polyfit(i[0], i[1], POLYNOMIAL_ORDER) for i in bucketed_features_long]
random.seed(900)
train_indx = random.sample(range(len(long_study_class)), 800)

train_X, test_X, train_Y, test_Y  = [], [], [], []
for i in range(len(long_study_class)):
    if i in train_indx:
        train_X.append(coeff_all[i])
        train_Y.append(long_study_class[i])
    else:
        test_X.append(coeff_all[i])
        test_Y.append(long_study_class[i])


gb = ensemble.GradientBoostingClassifier(n_estimators=5)
gb.fit(train_X, train_Y)
Yhat = gb.predict(train_X)
Yhat_test = gb.predict(test_X)
acc_test = accuracy(test_Y, Yhat_test)
acc_train = accuracy(train_Y, Yhat)

print "Training Data"
for i in range(10):
    print "\t Ploynomial Coefficients: [%.8f, %.4f, %.4f],  Label: %i"  %(train_X[i][0],train_X[i][1],train_X[i][2], train_Y[i])

print "Gradient Boosting: Order %i"  %POLYNOMIAL_ORDER
print "Test Accuracy: " + str(acc_test)
print "Train Accuracy: " + str(acc_train)


# knn = cluster.KMeans(n_clusters=4)
# knn.fit(train_X)
# Yhat = knn.predict(train_X)
# Yhat_test = knn.predict(test_X)
# acc_test_knn = accuracy(test_Y, Yhat_test)
# acc_train_knn = accuracy(train_Y, Yhat)
#
# print "KNN"
# print "Test Accuracy: " + str(acc_test_knn)
# print "Train Accuracy: " + str(acc_train_knn)
#
#
#
# for i in range(100):
#     print test_Y[i], Yhat_test[i]