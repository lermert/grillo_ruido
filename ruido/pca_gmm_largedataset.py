from Function_Clustering_DFs import run_pca, gmm
import numpy as np
from ruido.cc_dataset_mpi import CCDataset, CCData  # these are data structures which contain the array of cross-correlation traces, along with some metadata like the timestamps of each 
# cross-correlation window, the sampling rate, etc. The structures come with some basic processing functionality like filtering.
# CCDataset contains a dictionary with CCData, so that we can keep raw and stacked data by using different keys (dataset[0], dataset[1], etc.)
import os
from glob import glob
from obspy import UTCDateTime
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# input --------------------------------------------------------
# The input needed here are .h5 files with a specific structure
# providing correlation traces and some basic metadata (mostly,
# just the sampling rate and time stamps.)
# We could either adapt any data to be used in this script to 
# conform to this format, or adapt the CCDataset to accept
# a matrix of data as input too, and then edit lines 40-48 to 
# set that up.
input_directory = "../spectrograms/stacks_uwork"  #"/media/lermert/Ablage/corrs_from_workstation"  #
stations = ["UNM",]
ch1s = ["BHZ",]
ch2s = ["BHZ",]
fmins = [2.0, 4.0]
do_window = True
twin_mid = 0.0
filt_type = "bandpass"
# max. order: only relevant for cheby2_bandpass filter
filt_maxord = 18
lagmax = 100.0
# keep as many principal components as needed to explain this level of variance of the data
expl_var = 0.95
# nr. of random samples to be pulled from each file
n_samples_each_file = 1000  # int or "all"
# n clusters to try out; the final number will be based on estimating 
# the "knee" of the Bayesian information criterion (as implemented by Loic)
nclust = range(1, 13)
# end input -----------------------------------------------------


for ixsta, station in enumerate(stations):
    ch1 = ch1s[ixsta]
    ch2 = ch2s[ixsta]

    datafiles = glob(os.path.join(input_directory,
                                  "*{}*{}*{}.*.h5".format(station,
                                                          ch1,
                                                          ch2)))
    datafiles.sort()
    print(datafiles)

    dset = CCDataset(datafiles[0])
    dset.data_to_memory()

    # select a random subset of traces for PCA
    # here we use dataset key 0 for the raw data read in from each file
    # to retain all the randomly selected windows, we copy them to key 1
    if type(n_samples_each_file) == int:
        ixs_random = np.random.choice(np.arange(dset.dataset[0].ntraces),
                                  n_samples_each_file)
    elif n_samples_each_file == "all":
        ixs_random = range(dset.dataset[0].ntraces)
    dset.dataset[1] = CCData(dset.dataset[0].data[ixs_random].copy(),
                             dset.dataset[0].timestamps[ixs_random].copy(),
                             dset.dataset[0].fs)

    for ixfile, dfile in enumerate(datafiles):
        if ixfile == 0:
            # we've been here already
            continue
        # read the data in
        dset.add_datafile(dfile)
        dset.data_to_memory(keep_duration=0)
        # use a random subset of each file (unless "all" requested)
        if type(n_samples_each_file) == int:
            ixs_random = np.random.choice(np.arange(dset.dataset[0].ntraces),
                                  n_samples_each_file)
        elif n_samples_each_file == "all":
            ixs_random = range(dset.dataset[0].ntraces)
        newdata = dset.dataset[0].data[ixs_random, :]
        assert (newdata.base is not dset.dataset[0].data)

        # keep the randomly selected windows under key 1, adding to the previously selected ones
        dset.dataset[1].data = np.concatenate((dset.dataset[1].data,
                                              newdata))
        dset.dataset[1].timestamps = np.concatenate((dset.dataset[1].timestamps,
                                                     dset.dataset[0].timestamps[ixs_random]))
        dset.dataset[1].ntraces = dset.dataset[1].data.shape[0]


    for fmin in fmins:
        # The clustering is performed separately in different frequency bands. The selections may change depending on the frequency band.
        fmax = 2 * fmin
        twin_hw = 10. / fmin
        # filter before clustering
        dset.filter_data(stacklevel=1, filter_type=filt_type,
                         f_hp=fmin, f_lp=fmax, maxorder=filt_maxord)
        #window. The windows are all centered on lag 0 and extend to 10 / fmin
        dset.window_data(t_mid=twin_mid, hw=twin_hw,
                         window_type="tukey", tukey_alpha=0.5,
                         stacklevel=1, cutout=False)

        # perform PCA on the random subset
        dset.dataset[1].data = np.nan_to_num(dset.dataset[1].data)
        X = StandardScaler().fit_transform(dset.dataset[1].data)
        pca_rand = run_pca(X, min_cumul_var_perc=expl_var)
        # pca output is a scikit learn PCA object
        # just for testing, run the Gaussian mixture here
        # gm = gmm(pca_rand.transform(X), range(1, 12))

        all_pccs = []
        all_timestamps = []

        # now go through all files again, read, filter, window, and fit the pcs
        for datafile in datafiles:
            print(datafile)
            dset.add_datafile(datafile)
            dset.data_to_memory(keep_duration=0)
            dset.filter_data(stacklevel=0, filter_type=filt_type,
                             f_hp=fmin, f_lp=fmax, maxorder=filt_maxord)
            dset.window_data(t_mid=twin_mid, hw=twin_hw,
                             window_type="tukey", tukey_alpha=0.5,
                             stacklevel=0, cutout=False)
            dset.dataset[0].data = np.nan_to_num(dset.dataset[0].data)
            X = StandardScaler().fit_transform(dset.dataset[0].data)
            # expand the data in the principal component basis:
            pca_output = pca_rand.transform(X)
            # append to the list
            all_pccs.extend(pca_output)
            all_timestamps.extend(dset.dataset[0].timestamps)
        all_pccs = np.array(all_pccs)
        all_timestamps = np.array(all_timestamps)

        # do the clustering
        gmmodels, n_clusters, gmixfinPCA, probs, BICF = gmm(all_pccs, nclust)
        print(n_clusters, np.unique(gmixfinPCA))
        # save the cluster labels
        labels = np.zeros((2, len(all_timestamps)))
        labels[0] = all_timestamps
        labels[1] = gmixfinPCA
        outputfile = "{}.{}.{}.{}-{}Hz.gmmlabels.npy".format(station, ch1, ch2, fmin, fmax)
        np.save(outputfile, labels)

