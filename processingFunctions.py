import numpy as np
import time
from matplotlib import pyplot as plt


def separate_trajectories(raw_data):
    '''
    :param raw_data:
    :return: [
                [ [ID1, time after injury 1, TBI class, features], [ID1,time after injury 2, TBI class, features ], ["], ["] ],
                [ [ID2, time after injury 1, TBI class, features], [ID2,time after injury 2, TBI class, features ], ["][, "] ],
                ...
            ]
    Separates all observations into lists of common ID#s.  (i.e. a list of feature trajectories for each ID#)
    '''

    features = [[int(el.strip()) for el in row[10:len(row) - 1]] for row in raw_data]
    # Create a list of (ID , epoch in hrs, TBI class) tuples
    id_date = [[i[0].strip(),
                (int(time.mktime(time.strptime(i[1].strip(), '%Y-%m-%d'))) - int(
                    time.mktime(time.strptime(i[2].strip(), '%Y-%m-%d')))) / 3600.0,
                int(i[8])
                ] for i in raw_data]

    id_features = [id_date[i] + features[i] for i in range(len(id_date))]

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
    return trajectories


def get_feature_sum_lists(trajectories):
    '''
    :param trajectories:
    :return:
            [ TBI_class_0, TBI_class_1, ... , TBI_class_n ]
    :return:
            [
                [ (time_0, time_1, ... , time_n), (feature_sum_0, feature_sum_1, ... , feature_sum_n) ],  # for ID1
                [ (time_0, time_1, ... , time_n), (feature_sum_0, feature_sum_1, ... , feature_sum_n) ],  # for ID2
                ...
            ]
    '''
    TBI_class = [ob[0][2] for ob in trajectories]
    feature_sums = [[[ob[1], sum(ob[3:])] for ob in trajectory] for trajectory in trajectories]
    feature_sums_lists = np.array([zip(*el) for el in feature_sums])
    return feature_sums_lists, TBI_class


def separate_by_length(TBI_class, feature_sums):
    '''
    :param TBI_class:
    :param feature_sums:
    :return:
        (Feature_sum's that have more than 10 observation.)
    :return:
        (TBI_class's that have more than 10 observation.)
    :return:
        (Feature_sum's that have less than 10 observation.)
    :return:
        (TBI_class's that have less than 10 observation.)
    '''

    long_study = []
    long_study_class = []

    short_study = []
    short_study_class = []

    for i in range(len(feature_sums)):
        if len(feature_sums[i][0]) > 10:
            long_study.append(feature_sums[i])
            long_study_class.append(TBI_class[i])
        else:
            short_study.append(feature_sums[i])
            short_study_class.append(TBI_class[i])

    return long_study, long_study_class, short_study, short_study_class


def to_buckets1(feature_sums, nbins=5, evaluate=False):
    '''
    :param feature_sums:
    :param nbins:
    :return:
    (Feature_sums are bucketed into evenly spaced time intervals. The mean of feature_sums that occur in each time buckets are used)
    '''

    max_val = max([max(id[0]) for id in feature_sums])

    bins = np.linspace(0, max_val, nbins)
    bucketed_feature_sums = []
    bin_lengths = []
    for id in feature_sums:
        time_data = np.array(id[0])
        feature_data = np.array(id[1])
        digitized = np.digitize(time_data, bins)

        ###
        if evaluate:
            bin_length = [len(feature_data[digitized == i]) if len(feature_data[digitized == i]) > 0 else 0 for i in
                          range(1, len(bins))]
            bin_lengths.append(bin_length)
            avg_compression = np.mean(bin_lengths, axis=0)
        ###

        bin_means = [np.mean(feature_data[digitized == i]) if len(feature_data[digitized == i]) > 0 else 0 for i in
                     range(1, len(bins))]
        bucketed_feature_sums.append([tuple(bins[1:]), tuple(bin_means)])
    if evaluate:
        return bucketed_feature_sums, avg_compression
    else:
        return bucketed_feature_sums


def plot_trajectories(feature_sums_lists, TBI_class, obs=1000, length_of_study='long', comments='Continuous'):
    sample_size = len(TBI_class)

    if obs > sample_size:
        obs = sample_size

    if length_of_study == 'long':
        xlim = 9000
        study_type = 'Long'
    else:
        xlim = 1500
        study_type = 'Short'

    if len(TBI_class) != len(feature_sums_lists):
        return "Error: Class list is different lengths than feature_sum list."

    title = "%s Study: Random sample of %i IDs (out of %i)" % (study_type, obs, sample_size)

    fig = plt.figure()
    plt.xlim(0, xlim)
    plt.ylim(0, 100)

    cols = []
    for i in np.random.choice(range(len(TBI_class)), size=obs, replace=False):
        if TBI_class[i] == 0:
            col = 'black'
        elif TBI_class[i] == 1:
            col = 'red'
        elif TBI_class[i] == 2:
            col = 'green'
        else:
            col = 'blue'

        if col in set(cols):
            plt.plot(feature_sums_lists[i][0], feature_sums_lists[i][1], color=col)
        else:
            plt.plot(feature_sums_lists[i][0], feature_sums_lists[i][1], color=col, label=TBI_class[i])
            cols.append(col)

    fig.suptitle(title)
    plt.legend(loc='upper right')
    plt.ylabel('Feature Sum')
    plt.xlabel('Time After "Injury"')
    fig.savefig('TBI_plots/TBI_plot:_(%s, %s).jpg' % (study_type, comments))
    plt.show()


def only_features(bucketed_features):
    return np.array((bucketed_features[:, 1]))


def only_times(bucketed_features):
    return np.array((bucketed_features[:, 0]))


def flatten(array):
    """
    :param array:
    :return: list of all individual values in list of lists in order
    """
    return [i for sublist in array for i in sublist]


def separate_by_class(bucketed_features, TBI_class):
    long_study_class = np.array(TBI_class)
    TBI_0_features = bucketed_features[long_study_class == 0]
    TBI_1_features = bucketed_features[long_study_class == 1]
    TBI_2_features = bucketed_features[long_study_class == 2]
    TBI_3_features = bucketed_features[long_study_class == 3]
    return TBI_0_features, TBI_1_features, TBI_2_features, TBI_3_features


######## TESTING #########


def to_buckets_new(feature_sums, nbins=5, evaluate=False, technique="even"):
    '''
    :param feature_sums:
    :param nbins:
    :return:
    (Feature_sums are bucketed into evenly spaced time intervals. The mean of feature_sums that occur in each time buckets are used)
    '''

    max_val = max([max(id[0]) for id in feature_sums])

    if technique == "even":
        bins = np.linspace(0, max_val, nbins)
    else:
        min_length = min([len(i[0]) for i in feature_sums])
        diffs = [np.diff(i[0])[:min_length - 1] for i in feature_sums]
        mean_diffs = np.mean(diffs, axis=0)
        mean_diffs = list(mean_diffs) + [mean_diffs[-1] for i in range(5)]
        bins = [int(sum(mean_diffs[:i])) for i in range(0, len(mean_diffs), 4)] + [int(max_val)]

    bucketed_feature_sums = []
    bin_lengths = []
    for id in feature_sums:
        time_data = np.array(id[0])
        feature_data = np.array(id[1])
        digitized = np.digitize(time_data, bins)

        ###
        if evaluate:
            bin_length = [len(feature_data[digitized == i]) if len(feature_data[digitized == i]) > 0 else 0 for i in
                          range(1, len(bins))]
            bin_lengths.append(bin_length)
            avg_compression = np.mean(bin_lengths, axis=0)

        ###

        bin_means = [np.mean(feature_data[digitized == i]) if len(feature_data[digitized == i]) > 0 else 0 for i in
                     range(1, len(bins))]
        bucketed_feature_sums.append([tuple(bins[1:]), tuple(bin_means)])
    if evaluate:
        return bucketed_feature_sums, avg_compression
    else:
        return bucketed_feature_sums
