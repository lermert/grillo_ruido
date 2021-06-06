from Function_Clustering_DFs import run_pca, gmm
import numpy as np
from ruido.cc_dataset_mpi import CCDataset, CCData
import os
from glob import glob
from obspy import UTCDateTime
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# input --------------------------------------------------------
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
readstep = 3000
expl_var = 0.95
n_samples_each_file = 1000
nclust = range(1, 13)
# end input -----------------------------------------------------


for ixsta, station in enumerate(stations):
    ch1 = ch1s[ixsta]
    ch2 = ch2s[ixsta]

    datafiles = glob(os.path.join(input_directory,
                                  "*{}*{}*{}.*.h5".format(station,
                                                            ch1,
                                                            ch2)))
    print(datafiles)

    dset = CCDataset(datafiles[0])
    dset.data_to_memory()

    # select a random subset of traces for PCA
    ixs_random = np.random.choice(np.arange(dset.dataset[0].ntraces),
                                  n_samples_each_file)
    dset.dataset[1] = CCData(dset.dataset[0].data[ixs_random].copy(),
                             dset.dataset[0].timestamps[ixs_random].copy(),
                             dset.dataset[0].fs)

    for ixfile, dfile in enumerate(datafiles):
        if ixfile == 0:
            continue
        # read the data in
        dset.add_datafile(dfile)
        dset.data_to_memory(keep_duration=0)
        # use a random subset of each file
        ixs_random = np.random.choice(np.arange(dset.dataset[0].ntraces),
                                      n_samples_each_file)
        newdata = dset.dataset[0].data[ixs_random, :]
        assert (newdata.base is not dset.dataset[0].data)

        dset.dataset[1].data = np.concatenate((dset.dataset[1].data,
                                              newdata))
        dset.dataset[1].timestamps = np.concatenate((dset.dataset[1].timestamps,
                                                     dset.dataset[0].timestamps[ixs_random]))
        dset.dataset[1].ntraces = dset.dataset[1].data.shape[0]


    for fmin in fmins:
        fmax = 2 * fmin
        twin_hw = 10. / fmin
        # filter before clustering
        dset.filter_data(stacklevel=1, filter_type=filt_type,
                         f_hp=fmin, f_lp=fmax, maxorder=filt_maxord)
        #window
        dset.window_data(t_mid=twin_mid, hw=twin_hw,
                         window_type="tukey", tukey_alpha=0.5,
                         stacklevel=1, cutout=False)

        # perform PCA on this random subset
        X = StandardScaler().fit_transform(dset.dataset[1].data)
        pca_rand = run_pca(X, min_cumul_var_perc=expl_var)
        # pca output is an array with shape (nsamples_per_file * nr. of files, n pcas)
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
            X = StandardScaler().fit_transform(dset.dataset[0].data)
            pca_output = pca_rand.transform(X)

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

    # for fmin in fmins:
    #     dset.dataset[1] = CCData(data=dset.dataset[0].data.copy(),
    #                              timestamps=dset.dataset[0].timestamps.copy(),
    #                              fs=dset.dataset[0].fs)

    #     fmax = 2 * fmin
    #     twin_hw = 10. / fmin
    #     # filter - a little bit or a lot? take these features in one frequency band?
    #     dset.filter_data(stacklevel=1, filter_type="cheby2_bandpass",
    #                      f_hp=fmin, f_lp=fmax, maxorder=filt_maxord)

    #     # window if wanted
    #     # use tukey
    #     if do_window:
    #         dset.window_data(t_mid=twin_mid, hw=twin_hw,
    #                          window_type="tukey", tukey_alpha=0.5,
    #                          stacklevel=1, cutout=True)

    #     print(dset.dataset[1].data.shape)
    #     # standardize the data
    #     X = StandardScaler().fit_transform(dset.dataset[1].data)
    #     #  def Clustering_PCA_GMM(mat, PC_nb, range_GMM):
    #     pca_output, var, models, n_clusters, gmixfinPCA, probs, BICF = Clustering_PCA_GMM(X, nclust, 
    #         min_cumul_var_perc=expl_var)
    
