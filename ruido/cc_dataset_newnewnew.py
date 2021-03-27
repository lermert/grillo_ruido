import numpy as np
import h5py
from obspy import Trace, UTCDateTime
from obspy.signal.invsim import cosine_taper
import matplotlib.pyplot as plt
from scipy.signal import sosfilt, sosfiltfilt, hann, tukey
from scipy.fftpack import next_fast_len
from scipy.interpolate import interp1d
from ruido.utils import filter
import pandas as pd
import os
from noisepy.noise_module import stretching, dtw_dvv,\
stretching_vect, wts_dvv, whiten, mwcs_dvv, robust_stack
from ruido.clustering import cluster, cluster_minibatch
from obspy.signal.filter import envelope
from obspy.signal.detrend import polynomial as obspolynomial
from multiprocessing import Pool
# try:
#     import threading
# except:
#     pass


# class FilterThread (threading.Thread):
#     def __init__(self, traces, sos, zerophase, taper):
#         threading.Thread.__init__(self)
#         self.traces = traces
#         self.taper = taper
#         self.sos = sos
#         self.zp = zerophase
#     def run(self):
#         for i, tr in enumerate(self.traces):
#             if self.zp:
#                 self.traces[i, :] = sosfiltfilt(self.sos,
#                     self.taper * tr, padtype="even")
#             else:
#                 to_filter[i, :] = sosfilt(self.sos, self.taper * tr)

class FiltPool(object):
    def __init__(self, sos, taper, zerophase):
        self.sos = sos
        self.taper = taper
        self.zp = zerophase
    def run(self, trace):
        if self.zp:
            trace = sosfiltfilt(self.sos, self.taper * trace)
        else:
            trace = sosfilt(self.sos, self.taper * trace)

        
class CCData(object):
    """
    Class for keeping cross-correlation data and stacks of cross-correlation data
    """

    def __init__(self, data, timestamps, fs, kmeans_labels=None):

        self.data = np.array(data)
        self.timestamps = np.array(timestamps)
        self.fs = fs
        self.ntraces = self.data.shape[0]
        self.npts = self.data.shape[1]
        self.max_lag = (self.npts - 1) / 2 / self.fs
        self.lag = np.linspace(-self.max_lag, self.max_lag, self.npts)
        if kmeans_labels is not None:
            self.kmeans_labels = np.array(kmeans_labels)
        else:
            self.kmeans_labels = None
        self.add_rms()
        self.median = np.nanmedian(self.data, axis=0)

    def lose_allzero_windows(self):
        datanew = []
        tnew = []
        kmnew = []
        for i, d in enumerate(self.data):
            if d.sum() == 0.0:
                continue
            else:
                datanew.append(d)
                tnew.append(self.timestamps[i])
                if self.kmeans_labels is not None:
                    kmnew.append(self.kmeans_labels[i])

        self.data = np.array(datanew)
        self.timestamps = np.array(tnew)
        if self.kmeans_labels is not None:
            self.kmeans_labels = np.array(kmnew)
        self.ntraces = self.data.shape[0]
        self.add_rms()
        self.median = np.nanmedian(self.data, axis=0)

    def extend(self, new_data, new_timestamps,
               new_fs, keep_duration=0):
        # to do:
        # check compatibility of sampling rates and npts
        if new_fs != self.fs:
            raise ValueError("Cannot extend because old and new sampling rates do not match.")
        if self.kmeans_labels is not None:
            raise ValueError("Datasets with assigned clusters should not be extended. Build full dataset first, then cluster.")
        # find indices to keep
        if keep_duration >= 0:
            ixs_t_keep = np.where((self.timestamps[-1] - self.timestamps) <= keep_duration)[0]
        else:  # keep all if negative keep_duration
            ixs_t_keep = np.arange(len(self.timestamps))

        data = [d for d in self.data[ixs_t_keep]]
        timestamps = [t for t in self.timestamps[ixs_t_keep]]

        # overwrite data by new and old[from index onwards] data together
        data.extend(new_data)
        self.data = np.array(data)
        # overwrite timestamps by new and old[from index onwards] data together
        timestamps.extend(new_timestamps)
        self.timestamps = np.array(timestamps)
        # overwrite ntraces by new and old[from index onwards] data together
        self.ntraces = len(self.timestamps)
        self.rms = self.add_rms()
        self.median = np.nanmedian(self.data, axis=0)

    def add_rms(self):
        # add root mean square of raw cross-correlation windows
        # (for selection)
        rms = np.zeros(self.ntraces)

        if len(rms) == 0:
            return

        for i, dat in enumerate(self.data):
            rms[i] = np.sqrt(np.sum((dat - dat.mean()) ** 2) / len(dat))
           
        self.rms = rms



class CCDataset(object):
    """Processing object for Cross-correlation dataset organized by time stamp in hdf5 file."""

    def __init__(self, inputfile, ref=None, exclude_tag=None):
        """
        :type inputfile: string
        :param inputfile: File path to hdf5 file
        :type ref: numpy.ndarray or None
        :param ref: reference trace, e.g. from another file

        """
        super(CCDataset, self).__init__()

        self.station_pair = os.path.splitext(os.path.basename(inputfile))[0]
        self.station_pair = os.path.splitext(self.station_pair)[0]
        self.station_pair = os.path.splitext(self.station_pair)[0]
        self.datafile = h5py.File(inputfile, 'r')


        # self.add_dataset(inputfile)
        
        #self.stacklevels = {}
        self.dataset = {}
        # self.stacks = {}
        # self.stacks_timestamps = {}
        # self.stacks_lag = {}
        # self.stacks_kmeans_labels = {}


    def add_datafile(self, inputfile):

        self.datafile.close()
        self.station_pair = os.path.splitext(os.path.basename(inputfile))[0]
        self.station_pair = os.path.splitext(self.station_pair)[0]
        self.station_pair = os.path.splitext(self.station_pair)[0]
        self.datafile = h5py.File(inputfile, 'r')
        # fs = dict(self.datafile['stats'].attrs)['sampling_rate']
        # #self.delta = 1. / self.fs

        # if "data" in list(self.datafile["corr_windows"].keys()):
        #     npts = self.datafile['corr_windows']["data"][0].shape[0]
        # else:  #  old file format
        #     npts = self.datafile['corr_windows'][next(iter(self.datafile["corr_windows"]))].shape[0]
        # self.max_lag = (self.npts - 1) / 2 / self.fs

        #self.lag = np.linspace(-self.max_lag, self.max_lag, self.npts)

        # self.data = None
        # self.datakeys = []
        # self.timestamps = []
        # self.ref = np.zeros(self.npts)
        # self.rms = None
        # self.median = None
    def stacks_to_memory(self, kmeans_label=None):
        if len(self.dataset) == 0:
            ix_dataset = 1
        else:
            ix_dataset = max(list(self.dataset.keys())) + 1

        stacks = self.datafile["corr_stacks"]["data"][:]
        tstamps = self.datafile["corr_stacks"]["timestamps"][:]
        fs = self.datafile["stats"].attrs["sampling_rate"]
        kmeans_labels = None
        try:
            kmeans_labels = self.datafile["corr_stacks"]["kmeans_labels"][:]
        except KeyError:
            pass

        if kmeans_labels is not None and kmeans_label is not None:
            ixs = np.where(kmeans_labels == kmeans_label)[0]
            self.dataset[ix_dataset] = CCData(stacks[ixs], tstamps[ixs], fs)
        else:
            self.dataset[ix_dataset] = CCData(stacks, tstamps, fs)
            if kmeans_labels is not None:
                self.dataset[ix_dataset].kmeans_labels = kmeans_labels

    def __str__(self):
        output = ""
        output += "Cross-correlation dataset: {}\n".format(self.station_pair)
        #output += "Maximum lag: {} seconds\n".format(self.max_lag)
        #output += "Sampling rate: {} Hz\n".format(self.fs)
        #output += "Number of traces: {}\n".format(self.ntraces)
        for (k, v) in self.dataset.items():
            output += "Contains {} traces on stacking level {}\n".format(v.ntraces, k)
            output += "Starting {}, ending {}\n".format(UTCDateTime(v.timestamps[0]).strftime("%d.%m.%Y"),
                                                        UTCDateTime(v.timestamps[-1]).strftime("%d.%m.%Y"))
        # try:
        #     output += "Start date: {}\n".format((self.datafile["corr_windows"]["timestamps"][0]))
        #     i = -1
        #     enddate = ""
        #     while enddate == "":
        #         i -= 1
        #         enddate = self.datafile["corr_windows"]["timestamps"][i]
        #     output += "End date: {}\n".format(enddate)
        # except KeyError:
        #     pass
        return(output)

    def data_to_envelope(self, stacklevel=1):
        #replace stacks by their envelope

        newstacks = []
        for s in self.dataset[stacklevel].data:
            newstacks.append(envelope(s))
        self.dataset[stacklevel].data = np.array(newstacks)

    def data_to_memory(self, n_corr_max=None, t_min=None, t_max=None, keep_duration=0,
                       normalize=False):
        # read data from correlation file into memory and store in dataset[0]
        # use self.datafile
        # get fs, data, timestamps
        # commit to a new dataset object or add it to existing
        fs = dict(self.datafile['stats'].attrs)['sampling_rate']
        if "data" in list(self.datafile["corr_windows"].keys()):
            npts = self.datafile['corr_windows']["data"][0].shape[0]
            ntraces = len(np.where(self.datafile["corr_windows"]["timestamps"][:] != "")[0])
        else:  #  old file format
            npts = self.datafile['corr_windows'][next(iter(self.datafile["corr_windows"]))].shape[0]
            ntraces = len(self.datafile["corr_windows"])

        if n_corr_max is None:
            n_corr_max = ntraces

        # allocate data
        try:
            data = np.zeros((n_corr_max, npts))
        except MemoryError:
            print("Data doesn't fit in memory, set a lower n_corr_max")
            return()
        
        # allocate timestamps array
        timestamps = np.zeros(n_corr_max)
        try:  # new file format
            for i, v in enumerate(self.datafile["corr_windows"]["data"][:]):
                if i == n_corr_max:
                    break

                data[i, :] = v[:]
                #data.append(v)
                tstamp = self.datafile["corr_windows"]["timestamps"][i]
                #self.datakeys[i] = tstamp

                tstmp = '{},{},{},{},{}'.format(*tstamp.split('.')[0: 5])
                timestamps[i] = UTCDateTime(tstmp).timestamp
                if t_max is not None and tstmp > t_max:
                    break
                #timestamps.append(UTCDateTime(tstmp).timestamp)
        except KeyError:  # old file format
            for i, (k, v) in enumerate(self.datafile["corr_windows"].items()):
                if i == n_corr_max:
                    break
                data[i, :] = v[:]
                tstmp = '{},{},{},{},{}'.format(*k.split('.')[0: 5])
                timestamps[i] = UTCDateTime(tstmp).timestamp


        if t_min is not None:
            ix0 = np.argmin(abs(timestamps - t_min))
        else:
            ix0 = 0
        if t_max is not None:
            ix1 = np.argmin(abs(timestamps - t_max))
        else:
            ix1 = ntraces

        data = data[ix0: ix1, :]
        timestamps = timestamps[ix0: ix1]
        if normalize:
            # absolute last-resort-I-cannot-reprocess way to deal with amplitude issues
            for tr in data:
                tr /= tr.max()

        ntraces = ix1 - ix0
        print("Read to memory from {} to {}".format(UTCDateTime(timestamps[0]),
                                                    UTCDateTime(timestamps[-1])))

        if 0 in list(self.dataset.keys()):
            self.dataset[0].extend(data, timestamps, fs, keep_duration=keep_duration)
        else:
            self.dataset[0] = CCData(data, timestamps, fs)



#     def data_to_memory_prestack(self, n_corr_max=None, t_min=None, t_max=None, prestackstep=86400.):

#         if n_corr_max is None:
#             n_corr_max = self.ntraces

#         # estimate the required memory to avoid running into a memory error along the way
#         # if tmin, tmax given: estimate from there
#         # if not given: estimate from input file
#         if t_min is not None:
#             tm = t_min
#         else:
#             try:
#                 tmf = UTCDateTime(self.datafile["corr_windows"]["timestamps"][0]).timestamp
#             except KeyError:  # old file format
#                 tmf = UTCDateTime(list(self.datafile["corr_windows"])[0]).timestamp
#         # make sure only windows are used that actually are covered by the dataset
#         tm = max(tmf, tm)

#         if t_max is not None:
#             tma = t_max
#         else:
#             enddate = ""
#             try:
#                 i = -1
#                 while enddate == "":
#                     enddate = self.datafile["corr_windows"]["timestamps"][i]
#                     i -= 1
                
#             except KeyError:  # old file format
#                 i = -1
#                 while enddate == "":
#                     enddate = list(self.datafile["corr_windows"])[i]
#                     i -= 1

#             tma = UTCDateTime(enddate).timestamp

#         est_shape = (tma - tm) / prestackstep

#         try:
#             self.data = np.zeros((dat_shape, self.npts))
#         except MemoryError:
#             print("Data doesn't fit in memory, try setting a lower n_corr_max\
# or a higher prestack")
#             return()

#         # now, read in the data
#         self.data = []
#         self.datakeys = []  # np.zeros(dat_shape, dtype=np.str)
#         self.timestamps = []  # np.zeros(dat_shape)
#         current_stack = []
#         t_running = tm
#         while t_running <= tma:
#             try:  # new file format
#                 for i, v in enumerate(self.datafile["corr_windows"]["data"][:]):
#                     if i == n_corr_max:
#                         break

#                     # if current time - t_running: stack collected windows, append
#                     # the data and the timestamps

#                     if self.datakeys[ix] == "":
#                         tstamp = self.datafile["corr_windows"]["timestamps"][i]
#                         self.datakeys[ix] = tstamp

#                         tstamp = '{},{},{},{},{}'.format(*tstamp.split('.')[0: 5])
#                         self.timestamps[ix] = UTCDateTime(tstamp).timestamp

#             except KeyError:  # old file format
#                 for i, (k, v) in enumerate(self.datafile["corr_windows"].items()):
#                     if i == n_corr_max:
#                         break
#                     if ix == dat_shape:
#                         break
#                     ix = int(i // prestack)
#                     self.data[ix, :] += v[:] / prestack
#                     self.datakeys[ix] = k
#                     tstamp = '{},{},{},{},{}'.format(*k.split('.')[0: 5])
#                     self.timestamps[ix] = UTCDateTime(tstamp).timestamp

#         #self.datakeys = np.array(self.datakeys)
#         #self.timestamps = np.array(self.timestamps)
#         self.ncorr = 
#         self.data = self.data[ix0: ix1]
#         self.timestamps = self.timestamps[ix0: ix1]
#         self.datakeys = self.datakeys[ix0: ix1]
#         self.ntraces = ix1 - ix0
#         print("Read to memory from {} to {}".format(UTCDateTime(self.timestamps[0]),
#                                                     UTCDateTime(self.timestamps[-1])))


    

    def select_by_percentile(self, ixs, stacklevel=0, perc=90, measurement="RMS",
                             mode="upper", debug_mode=False):
        # select on the basis of relative root mean square amplitude
        # of the cross-correlations

        if self.dataset[stacklevel].rms is None:
            self.dataset[stacklevel].add_rms()
        rms = self.dataset[stacklevel].rms[ixs]
        if mode == "upper":
            ixs_keep = np.where(rms <= np.nanpercentile(rms, perc))
        elif mode == "lower":
            ixs_keep = np.where(rms >= np.nanpercentile(rms, perc))
        elif mode == "both":
            ixs_keep = np.intersect1d(np.where(rms <= np.nanpercentile(rms, perc)),
                                      np.where(rms >= np.nanpercentile(rms, 100 - perc)))
        if debug_mode:
            print("Selection by percentile of RMS: Before, After", len(ixs), len(ixs_keep))

        return(ixs[ixs_keep])


    def group_for_stacking(self, t0, duration, stacklevel=0, kmeans_label=None, bootstrap=0):
        """
        Create a list of time stamps to be stacked
        --> figure out what "belongs together" in terms of time window
        afterwards, more elaborate selections can be applied :)
        Alternatively, group stacks together for forming longer stacks
        e.g. after clustering
        """

        t_to_select = self.dataset[stacklevel].timestamps

        # find closest to t0 window
        assert type(t0) in [float, np.float64], "t0 must be floating point time stamp"

        # find indices
        # longer_zero = np.array([len(d) > 0 for d in self.dataset[stacklevel].data])
        ixs_selected = np.intersect1d(np.where(t_to_select >= t0),
                                      np.where(t_to_select < (t0 + duration)))
                                      # np.where(longer_zero))

        # check if selection to do for clusters
        if kmeans_label is not None:
            k_to_select = self.dataset[stacklevel].kmeans_labels
            if k_to_select is None:
                raise ValueError("Selection by cluster labels not possible: No labels assigned.")
            ixs_selected = np.intersect1d(ixs_selected, np.where(k_to_select == kmeans_label))

        if bootstrap > 0:
            ixs_out = []
            for i in range(bootstrap):
                ixs_out.extend(np.random.choice(ixs_selected, len(ixs_selected)))
        else:
            ixs_out = ixs_selected

        return(ixs_out)

    def select_for_stacking(self, ixs, selection_mode, stacklevel=0, bootstrap=False, cc=0.5,
                            twin=None, ref=None, dist=None, distquant=None, nr_bootstrap=1, **kwargs):
        """
        Select by: closeness to reference or percentile or...
        Right now this only applies to raw correlations, not to stacks
        """

        lag = self.dataset[stacklevel].lag
        data = self.dataset[stacklevel].data
        if len(ixs) == 0:
            return([])

        if selection_mode == "rms_percentile":
            ixs_selected = self.select_by_percentile(ixs, stacklevel, **kwargs)

        elif selection_mode == "cc_to_median":
            median = self.dataset[stacklevel].median
            ixs_selected = []
            if twin is not None:
                cc_ixs = [np.argmin((lag - t) ** 2) for t in twin]
            else:
                cc_ixs = [0, -1]
            for i in ixs:
                corrcoeff = np.corrcoef(data[i, cc_ixs[0]: cc_ixs[1]], 
                                        median[cc_ixs[0]: cc_ixs[1]])[0, 1]
                if corrcoeff < cc or np.isnan(corrcoeff):
                    continue
                ixs_selected.append(i)
            ixs_selected = np.array(ixs_selected)

        elif selection_mode == "cc_to_ref":
            ixs_selected = []
            if ref is None:
                raise ValueError("Reference must be given.")
            if twin is not None:
                cc_ixs = [np.argmin((lag - t) ** 2) for t in twin]
            else:
                cc_ixs = [0, -1]
            for i in ixs:
                corrcoeff = np.corrcoef(data[i, cc_ixs[0]: cc_ixs[1]],
                                        ref[cc_ixs[0]: cc_ixs[1]])[0, 1]
                if corrcoeff < cc or np.isnan(corrcoeff):
                    continue
                ixs_selected.append(i)
            ixs_selected = np.array(ixs_selected)
        elif selection_mode == "distance_to_cent":
            # get the distances for this particular cl
            ixs_selected = ixs[np.where(dist <= distquant)]

        else:
            raise NotImplementedError

        if bootstrap:
            ixs_sel = np.zeros((nr_bootstrap, len(ixs_selected)))
            for i in range(nr_bootstrap):
                ixs_sel[i, :] = np.random.choice(ixs_selected, len(ixs_selected))  # sampling with replacing
            ixs_selected = ixs_sel
        return(ixs_selected)

    def stack(self, ixs, stackmode="linear", stacklevel_in=0, stacklevel_out=1, overwrite=False,
              epsilon_robuststack=None):
        #stack
        if len(ixs) == 0:
            return()

        #newstacks = []
        #newt = []
        
        to_stack = self.dataset[stacklevel_in].data
        t_to_stack = self.dataset[stacklevel_in].timestamps.copy()

        if stackmode == "linear":
            s = to_stack[ixs].sum(axis=0)
            newstacks = s / len(ixs)
            newt = t_to_stack[ixs[0]]
        elif stackmode == "median":
            newstacks = np.median(to_stack[ixs], axis=0)
            newt = t_to_stack[ixs[0]]
        elif stackmode == "robust":
            newstacks, w, nstep = robust_stack(to_stack[ixs], epsilon_robuststack)
            print(newstacks.shape, " NEWSTACKS ", nstep)
            newt = t_to_stack[ixs[0]]
        else:
            raise ValueError("Unknown stacking mode {}".format(stackmode))
        # self.stacks[self.timestamps[ixs[0]]] = s

        #i#f self.stacks is None:
         #   self.stacks_timestamps = np.array(newt)
         #   self.stacks = np.array(newstacks)
        #else:
         #   temp = self.stacks.copy()
        #self.stacks[0: temp.shape[0]] = temp
        #self.stacks[temp.shape[0]: ] = np.array(newstacks)[:]

        #tempt = self.stacks_timestamps.copy()
        #self.stacks_timestamps[0: len(tempt)] = tempt
        #self.stacks_timestamps[len(tempt): ] = np.array(newt)
        # #if stacklevel_out not in list(self.dataset.keys()):
            # self.dataset[stacklevel_out] = CCData([newstacks], [newt], self.dataset[stacklevel_in].fs)
        try:  # elif stacklevel_out in list(self.dataset.keys()) and overwrite == False:
            self.dataset[stacklevel_out].extend([newstacks], [newt],
                                                self.dataset[stacklevel_in].fs, keep_duration=-1)
        except KeyError:
            self.dataset[stacklevel_out] = CCData([newstacks], [newt], self.dataset[stacklevel_in].fs)
        #elif stacklevel_out in list(self.dataset.keys()) and overwrite == True:
        #    self.dataset[stacklevel_out] = CCData(newstacks, newt, self.dataset[stacklevel_in].fs)

    def data_to_dataframe(self, lag0, lag1, stacklevel, normalize=False):

        to_put_in_dataframe = self.dataset[stacklevel].data 
        lag = self.dataset[stacklevel].lag
        nts = len(to_put_in_dataframe)

        ixs_get0 = np.where(lag >= lag0)
        ixs_lets = np.where(lag <= lag1)
        ixs = np.intersect1d(ixs_get0, ixs_lets)
        #self.df_lags = self.lag[ixs]
        nr_samples = len(ixs)

        self.df = pd.DataFrame(columns=range(nr_samples),
                               data=np.zeros((nts, nr_samples)))
        for i in range(nts):
            for j in range(nr_samples):
                if normalize:
                    self.df.iat[i, j] = to_put_in_dataframe[i, ixs][j] / (to_put_in_dataframe[i, :].max() + np.finfo(float).eps)
                else:
                    self.df.iat[i, j] = to_put_in_dataframe[i, ixs][j]

    # def stacks_to_dataframe(self, lag0, lag1, normalize=True):

    #     tstamps = np.array(list(self.stacks.keys()))
    #     ixs_get0 = np.where(self.lag >= lag0)
    #     ixs_lets = np.where(self.lag <= lag1)
    #     ixs = np.intersect1d(ixs_get0, ixs_lets)
    #     self.df_lags = self.lag[ixs]
    #     nr_samples = len(ixs)

    #     self.df = pd.DataFrame(columns=range(nr_samples),
    #                            data=np.zeros((len(self.stacks), nr_samples)))
    #     for i in range(len(self.stacks)):
    #         for j in range(nr_samples):
    #             if normalize:
    #                 self.df.iat[i, j] = self.data[i, ixs][j] / (self.data[i, ixs].max() + np.finfo(float).eps)
    #             else:
    #                 self.df.iat[i, j] = self.data[i, ixs][j]


    # def data_from_mem(self, ix=None, k=None):
    #     if ix is None:
    #         return self.data[self.datakeys.index(k), :]
    #     else:
    #         return self.data[ix, :]

    # def reference(self, reftype, t_min=None, t_max=None, overwrite=False):

    #     """
    #     t_min and t_max are start- and endtimes for forming a reference
    #     If None are given, then all windows are used.
    #     """

    #     if self.ref.sum() != 0.0 and not overwrite:
    #         raise ValueError("Reference was already given, not overwriting.")

    #     if False in [t_min is None, t_max is None]:
    #         if t_min is not None and t_max is None:
    #             ixs = np.where(np.array(self.timestamps) >= t_min)[0]
    #         elif t_min is None and t_max is not None:
    #             ixs = np.where(np.array(self.timestamps) <= t_max)[0]
    #         else:
    #             ixs = np.where(np.array(self.timestamps) >= t_min)[0]  
    #             ixs = np.where(np.array(self.timestamps)[ixs] <= t_max)[0]
    #     else:
    #         ixs = np.where(np.array(self.timestamps) > 0)[0]  # everywhere
        

    #     if reftype == "arithmetic_mean":
    #         if self.data is not None:
    #             self.ref = np.mean(self.data[ixs], axis=0)

    #     elif reftype == "median":
    #         if self.data is not None:
    #             self.ref = np.nanmedian(self.data[ixs], axis=0)


    # def filter_stacks(self, taper_perc=0.05, filter_type="bandpass",
    #                   f_hp=None, f_lp=None, corners=4, zerophase=True,
    #                   maxorder=8):
    #     if filter_type == "bandpass":
    #         for (k, tr) in self.stacks.items():
    #             tr.taper(taper_perc)
    #             tr.filter(filter_type, freqmin=f_hp,
    #                       freqmax=f_lp, corners=corners, zerophase=zerophase)
    #     elif filter_type == "cheby2_bandpass":
    #         sos = filter.cheby2_bandpass(df=self.fs, freq0=f_hp, freq1=f_lp,
    #                                      maxorder=maxorder)

    #         taper = cosine_taper(self.npts, taper_perc)
    #         for (k, tr) in self.stacks.items():
    #             if zerophase:
    #                 firstpass = sosfilt(sos, taper * tr.data)
    #                 tr.data = sosfilt(sos, firstpass[::-1])[::-1]
    #                 # then assign to newfile, which might be self.file
    #             else:
    #                 tr.data = sosfilt(sos, taper * tr.data)
    #     elif filter_type == "cwt":
    #         taper = cosine_taper(self.npts, taper_perc)
    #         for (k, tr) in self.stacks.items():
    #             tr.data = filter.cwt_bandpass(taper * tr.data, f_hp, f_lp, df=self.fs)

    def demean(self, stacklevel=0):

        to_demean = self.dataset[stacklevel].data
        for d in to_demean:
            d -= d.mean()

    def detrend(self, stacklevel=0, order=3):
        to_detrend = self.dataset[stacklevel].data
        for d in to_detrend:
            obspolynomial(d, order=order)

    # # To do: common filter function for data, stacks reference!!!
    # def filter_reference(self, taper_perc=0.05, filter_type="bandpass",
    #                      f_hp=None, f_lp=None, corners=4, zerophase=True,
    #                      maxorder=8):
    #     taper = cosine_taper(self.npts, taper_perc)
    #     if filter_type == 'bandpass':
    #         if None in [f_hp, f_lp]:
    #             raise TypeError("f_hp and f_lp (highpass and lowpass frequency) must be floats.")
    #         sos = filter.bandpass(df=self.fs, freqmin=f_hp, freqmax=f_lp,
    #                               corners=corners)
    #     elif filter_type == 'lowpass':
    #         sos = filter.lowpass(df=self.fs, freq=f_lp, corners=corners)
    #     elif filter_type == 'highpass':
    #         sos = filter.highpass(df=self.fs, freq=f_hp, corners=corners)
    #     elif filter_type == "cheby2_bandpass":
    #         sos = filter.cheby2_bandpass(df=self.fs, freq0=f_hp, freq1=f_lp,
    #                                      maxorder=maxorder)
    #     elif filter_type == "cwt":
    #         taper = cosine_taper(self.npts, taper_perc)
    #         self.ref = filter.cwt_bandpass(taper * self.ref, f_hp, f_lp, df=self.fs)
    #     else:
    #         msg = 'Filter %s is not implemented, implemented filters:\
    #         bandpass, highpass,lowpass' % type
    #         raise ValueError(msg)

    #     if filter_type != "cwt":
    #         if zerophase:
    #             firstpass = sosfilt(sos, taper * self.ref)
    #             self.ref = sosfilt(sos, firstpass[::-1])[::-1]
    #             # then assign to newfile, which might be self.file
    #         else:
    #             self.ref = sosfilt(sos, taper * self.ref)

    def filter_data(self, stacklevel=0, taper_perc=0.1, filter_type="bandpass",
                    f_hp=None, f_lp=None, corners=4, zerophase=True,
                    maxorder=8, npool=1):

        to_filter = self.dataset[stacklevel].data
        npts = self.dataset[stacklevel].npts
        fs = self.dataset[stacklevel].fs

        # check that the input array has 2 dimensions
        if not np.ndim(to_filter) == 2:
            raise ValueError("Input array for filtering must have dimensions of n_traces * n_samples")

        # define taper to avoid high-freq. artefacts
        taper = cosine_taper(npts, taper_perc)

        
        # define filter
        if filter_type == 'bandpass':
            if None in [f_hp, f_lp]:
                raise TypeError("f_hp and f_lp (highpass and lowpass frequency) must be floats.")
            sos = filter.bandpass(df=fs, freqmin=f_hp, freqmax=f_lp,
                                  corners=corners)
        elif filter_type == 'lowpass':
            sos = filter.lowpass(df=fs, freq=f_lp, corners=corners)
        elif filter_type == 'highpass':
            sos = filter.highpass(df=fs, freq=f_hp, corners=corners)
        elif filter_type == "cheby2_bandpass":
            sos = filter.cheby2_bandpass(df=fs, freq0=f_hp, freq1=f_lp,
                                         maxorder=maxorder)
        elif filter_type == "cwt":
            taper = cosine_taper(npts, taper_perc)
            for i, tr in enumerate(to_filter):
                to_filter[i, :] = filter.cwt_bandpass(tr, f_hp, f_lp, df=fs)
        else:
            msg = 'Filter %s is not implemented, implemented filters:\
            bandpass, highpass,lowpass' % type
            raise ValueError(msg)

        filtpool = FiltPool(sos, taper, zerophase)
        if filter_type != "cwt":
            #for i, tr in enumerate(to_filter):
            with Pool(npool) as p:
                p.map(filtpool.run, to_filter)
                # if zerophase:
                #     to_filter[i, :] = sosfiltfilt(sos, taper * tr, padtype="even")
                # else:
                #     to_filter[i, :] = sosfilt(sos, taper * tr)


    def get_window(self, t_mid, hw, lag, window_type, alpha=0.2):

        # half_lag_minus_one = int((self.npts - 1) / 2)
        ix_1 = np.argmin(abs(lag - t_mid - hw))
        ix_0 = np.argmin(abs(lag - t_mid + hw))  # [self.lag > 0]
        win = np.zeros(lag.shape)
        if window_type == "hann":
            # win[half_lag_minus_one + ix_0 : half_lag_minus_one + ix_1] = hann(ix_1 - ix_0)
            win[ix_0: ix_1] = hann(ix_1 - ix_0)
        elif window_type == "tukey":
            win[ix_0: ix_1] = tukey(ix_1 - ix_0, alpha=alpha)
        elif window_type == "boxcar":
            win[ix_0: ix_1] = 1.0
        return(win)

    def post_whiten(self, f1, f2, npts_smooth=5, stacklevel=0):

        nfft = int(next_fast_len(self.dataset[stacklevel].npts))
        td_taper = cosine_taper(self.dataset[stacklevel].npts, 0.1)
        fft_para = {"dt": 1./self.dataset[stacklevel].fs, 
                     "freqmin": f1,
                     "freqmax": f2,
                     "smooth_N": 5,
                     "freq_norm": "phase_only"}
        freq = np.fft.fftfreq(n=nfft,
                              d=1./self.dataset[stacklevel].fs)
        
        for i, tr in enumerate(self.dataset[stacklevel].data):
            spec = np.zeros(freq.shape)
            # spec = np.fft.rfft(td_taper * tr, n=2*self.dataset[stacklevel].npts)
            # nume = spec.copy() * taper
            # nume = self.moving_average(nume, n=npts_smooth)
            # spec /= nume
            # self.dataset[stacklevel].data[i, :] = td_taper * np.real(np.fft.irfft(spec,
            #                                       n=2*self.dataset[stacklevel].npts))[0: self.dataset[stacklevel].npts]
            spec = whiten(td_taper * tr, fft_para)
            self.dataset[stacklevel].data[i, :] = np.real(np.fft.ifft(spec, n=nfft)[0: self.dataset[stacklevel].npts])

    def moving_average(self, a, n=3):
        ret = np.cumsum(a, dtype=np.complex)
        ret[n:] = ret[n:] - ret[:-n]
        return ret / n

    def window_data(self, stacklevel, t_mid, hw,
                    window_type="hann", tukey_alpha=0.2,
                    cutout=False):
        # check that the input array has 2 dimensions
        to_window = self.dataset[stacklevel].data
        lag = self.dataset[stacklevel].lag
        if not np.ndim(to_window) == 2:
            raise ValueError("Input array for windowing must have dimensions of n_traces * n_samples")

        win = self.get_window(t_mid, hw, lag, window_type=window_type, alpha=tukey_alpha)

        if not cutout:
            for ix in range(to_window.shape[0]):
                to_window[ix, :] *= win
        else:
            new_win_dat = []
            for ix in range(to_window.shape[0]):
                ix_to_keep = np.where(win > 0.0)[0]
                new_win_dat.append(to_window[ix, ix_to_keep])
            newlag = self.dataset[stacklevel].lag[ix_to_keep]
            self.dataset[stacklevel].lag = newlag
            self.dataset[stacklevel].data = new_win_dat
            self.dataset[stacklevel].npts = len(ix_to_keep)

    def measure_dvv(self, ref, f0, f1, stacklevel=1, method="stretching",
                    ngrid=90, dvv_bound=0.03,
                    measure_smoothed=False, indices=None,
                    moving_window_length=None, slide_step=None, maxlag_dtw=0.0,
                    len_dtw_msr=None):

        to_measure = self.dataset[stacklevel].data
        lag = self.dataset[stacklevel].lag
        timestamps = self.dataset[stacklevel].timestamps
        # print(timestamps)
        fs = self.dataset[stacklevel].fs

        if len(to_measure) == 0:
            return()

        reference = ref.copy()
        para = {}
        para["dt"] = 1. / fs
        para["twin"] = [lag[0], lag[-1] + 1. / fs]
        para["freq"] = [f0, f1]

        if indices is None:
            indices = range(len(to_measure))

        dvv_times = np.zeros(len(indices))
        ccoeff = np.zeros(len(indices))
        best_ccoeff = np.zeros(len(indices))

        if method in ["stretching", "mwcs"]:
            dvv = np.zeros((len(indices), 1))
            dvv_error = np.zeros((len(indices), 1))
        elif method in ["cwt-stretching"]:
            if len_dtw_msr is None:
                testmsr = wts_dvv(reference, reference, True, 
                                  para, dvv_bound, ngrid)
                dvv = np.zeros((len(indices), len(testmsr[1])))
                dvv_error = np.zeros((len(indices), len(testmsr[1])))
            
        elif method in ["dtw"]:
            if len_dtw_msr is None:
                len_dtw_msr = []
                testmsr = dtw_dvv(reference, reference,
                              para, maxLag=maxlag_dtw,
                              b=10, direction=1)
                len_dtw_msr.append(len(testmsr[0]))
                len_dtw_msr.append(testmsr[1].shape)

            dvv = np.zeros((len(indices), len_dtw_msr[0]))
            dvv_error = np.zeros((len(indices), *len_dtw_msr[1]))
        else:
            raise ValueError("Unknown measurement method {}.".format(method))


        cnt = 0
        for i, tr in enumerate(to_measure):
            if i not in indices:
                continue

            if method == "stretching":
                dvvp, delta_dvvp, coeffp, cdpp = stretching_vect(reference, tr,
                                                        dvv_bound, ngrid, para)
                cwtfreqs = []
            elif method == "dtw":
                dvv_bound = int(dvv_bound)
                warppath, dist,  coeffor, coeffshift = dtw_dvv(reference, tr,
                                         para, maxLag=maxlag_dtw,
                                         b=dvv_bound, direction=1)
                coeffp = coeffshift
                cdpp = coeffor
                delta_dvvp = dist
                dvvp = warppath
                cwtfreqs = []
            elif method == "mwcs":
                ixsnonzero = np.where(reference != 0.0)
                dvvp, errp = mwcs_dvv(reference[ixsnonzero],
                                     tr[ixsnonzero],
                                     moving_window_length,
                                     slide_step, para)
                delta_dvvp = errp
                coeffp = np.nan
                cdpp = np.nan
                cwtfreqs = []
            elif method == "cwt-stretching":
                cwtfreqs, dvvp, errp = wts_dvv(reference, tr, True, 
                                               para, dvv_bound, ngrid)
                delta_dvvp = errp
                coeffp = np.nan
                cdpp = np.nan
                cwt_freqs = cwtfreqs

            dvv[cnt, :] = dvvp
            dvv_times[cnt] = timestamps[i]
            ccoeff[cnt] = cdpp
            # print(ccoeff[cnt])
            best_ccoeff[cnt] = coeffp
            dvv_error[cnt, :] = delta_dvvp
            cnt += 1
        return(dvv, dvv_times, ccoeff, best_ccoeff, dvv_error, cwtfreqs)

    def interpolate_stacks(self, new_fs, stacklevel=1):

        fs = self.dataset[stacklevel].fs
        max_lag = self.dataset[stacklevel].max_lag
        lag = self.dataset[stacklevel].lag
        stacks = self.dataset[stacklevel].data
        npts = self.dataset[stacklevel].data.shape[-1]


        if (new_fs % self.dataset[stacklevel].fs) != 0:
            raise ValueError("For the moment only integer-factor resampling is permitted.")

        new_npts = int((npts - 1.) / fs * new_fs) + 1
        print("new_npts: ", new_npts)
        new_lag = np.linspace(-max_lag, max_lag, new_npts)

        newstacks = []
        for stack in stacks:
            f = interp1d(lag, stack, kind="cubic")
            newstacks.append(f(new_lag))
        self.dataset[stacklevel].data = np.array(newstacks)

        # if self.ref is not None:
        #     f = interp1d(lag, self.ref, kind="cubic")
        #     self.ref = f(new_lag)
        self.dataset[stacklevel].lag = new_lag
        self.dataset[stacklevel].npts = new_npts
        self.dataset[stacklevel].fs = 1. / (new_lag[1] - new_lag[0])  #(npts - 1) / (lag[-1] - lag[0])

    def form_clusters(self, n_clusters, stacklevel=1,
                      lag0=0, lag1=20, n_corr_max=None,
                      normalize=False, method="kmeans",
                      n_iterations=100):

        self.data_to_dataframe(lag0=lag0, lag1=lag1, stacklevel=stacklevel, normalize=normalize)

        if method == "kmeans":
            kmeans_labels, centroids, inertia, n_it_cluster = cluster(self.df, n_clusters)

        elif method == "minibatch":
            kmeans_labels, centroids, inertia, n_it_cluster = cluster_minibatch(self.df, n_clusters, n_iterations=n_iterations)

        self.dataset[stacklevel].kmeans_labels = kmeans_labels

        # restore original shape and add centroids
        cents = []
        for i, c in enumerate(centroids):
            cent = np.zeros(self.dataset[stacklevel].npts)
            ixs_get0 = np.where(self.dataset[stacklevel].lag >= lag0)
            ixs_lets = np.where(self.dataset[stacklevel].lag <= lag1)
            ixs = np.intersect1d(ixs_get0, ixs_lets)
            cent[ixs] = c
            cents.append(cent)
        self.dataset[stacklevel].centroids = np.array(cents)

        # also add distance to centroid for each data
        dists = []
        for ix, d in enumerate(self.dataset[stacklevel].data):
            ix_k = self.dataset[stacklevel].kmeans_labels[ix]
            dist = (d / d.max() - cents[ix_k] / cents[ix_k].max()) ** 2
            dists.append(dist)
        self.dataset[stacklevel].kmeans_dist = np.array(dists)

        # append also the median, 75th and 90th percentile of distance
        # to centroid for each cluster
        distquant = np.zeros((n_clusters, 3))
        for i in range(n_clusters):
            ixs_cluster = np.where(self.dataset[stacklevel].kmeans_labels == i)
            med = np.median(self.dataset[stacklevel].kmeans_dist[ixs_cluster])
            nintieth = np.percentile(self.dataset[stacklevel].kmeans_dist[ixs_cluster], 90.)
            seventyfifth = np.percentile(self.dataset[stacklevel].kmeans_dist[ixs_cluster], 75.)
            distquant[i, 0] = med
            distquant[i, 2] = nintieth
            distquant[i, 1] = seventyfifth
        self.dataset[stacklevel].dist_quant = distquant
        return(inertia)

    def plot_stacks(self, stacklevel=1, outfile=None, seconds_to_show=20, scale_factor_plotting=0.1,
        plot_mode="heatmap", seconds_to_start=0.0, cmap=plt.cm.bone, mask_gaps=False, step=None, figsize=None,
        color_by_cc=False, normalize_all=False, label_style="month", ax=None, plot_envelope=False, ref=None,
        mark_17_quake=False):

        if mask_gaps and step == None:
            raise ValueError("To mask the gaps, you must provide the step between successive windows.")

        to_plot = self.dataset[stacklevel].data
        t_to_plot = self.dataset[stacklevel].timestamps
        lag = self.dataset[stacklevel].lag

        if to_plot.shape[0] == 0:
            return()
        ylabels = []
        ylabelticks = []
        months = []
        years = []

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax1 = fig.add_subplot(111)
        else:
            ax1 = ax

        if plot_mode == "traces":            
            cnt = 0
            for i, tr in enumerate(to_plot):

                if color_by_cc:
                    cmap = plt.cm.get_cmap("Spectral", 20)
                    crange = np.linspace(-1.0, 1.0, 20)
                    cmap = cmap(crange)
                    cccol = np.corrcoef(tr, ref)[0][1]
                    ix_colr = np.argmin((cccol - crange) ** 2)
                    ax1.plot(lag, tr / tr.max() + scale_factor_plotting * cnt,
                             c=cmap[ix_colr], alpha=0.5)
                else:
                    ax1.plot(lag, tr / tr.max() + scale_factor_plotting * cnt,
                             'k', alpha=0.5, linewidth=0.5)

                t = t_to_plot[i]
                if label_style == "month":
                    if UTCDateTime(t).strftime("%Y%m") not in months:
                        ylabels.append(scale_factor_plotting * cnt)
                        ylabelticks.append(UTCDateTime(t).strftime("%Y/%m/%d"))
                        months.append(UTCDateTime(t).strftime("%Y%m"))
                elif label_style == "year":
                    if UTCDateTime(t).strftime("%Y") not in years:
                        ylabelticks.append(UTCDateTime(t).strftime("%Y./%m/%d"))
                        ylabels.append(scale_factor_plotting * cnt)
                        years.append(UTCDateTime(t).strftime("%Y"))
                cnt += 1

            if ref is not None:
                ax1.plot(lag, ref / ref.max(), linewidth = 2.0)
                ax1.set_ylim([-1, scale_factor_plotting * cnt + np.mean(abs(tr))])
            else:
                ax1.set_ylim([0 - np.mean(abs(tr)), 
                             scale_factor_plotting * cnt + np.mean(abs(tr))])

        elif plot_mode == "heatmap":

            if not mask_gaps:
                dat_mat = np.zeros((self.dataset[stacklevel].ntraces, self.dataset[stacklevel].npts))
                for ix, tr in enumerate(to_plot):
                    if normalize_all:
                        dat_mat[ix, :] = tr / tr.max()
                    else:
                        dat_mat[ix, :] = tr
                    if plot_envelope:
                        dat_mat[ix, :] = envelope(dat_mat[ix, :])
                    t = t_to_plot[ix]
                    if label_style == "month":
                        if UTCDateTime(t).strftime("%Y%m") not in months:
                            ylabels.append(t)
                            ylabelticks.append(UTCDateTime(t).strftime("%Y/%m/%d"))
                            months.append(UTCDateTime(t).strftime("%Y%m"))

                    elif label_style == "year":
                        if UTCDateTime(t).strftime("%Y") not in years:
                            ylabels.append(t)
                            ylabelticks.append(UTCDateTime(t).strftime("%Y/%m/%d"))
                            years.append(UTCDateTime(t).strftime("%Y"))

                        
                if plot_envelope:
                    vmin = 0
                    vmax = scale_factor_plotting * dat_mat.max()
                else:
                    vmin = -scale_factor_plotting * dat_mat.max()
                    vmax = scale_factor_plotting * dat_mat.max() 

            else:
                tstamp0 = t_to_plot[0]
                tstamp1 = t_to_plot[-1]
                t_to_plot_all = np.arange(tstamp0, tstamp1 + step, step=step)
                dat_mat = np.zeros((len(t_to_plot_all), self.dataset[stacklevel].npts))
                dat_mat[:, :] = np.nan
        
                for ix, tr in enumerate(to_plot):
                    t = t_to_plot[ix]
                    ix_t = np.argmin(np.abs(t_to_plot_all - t))
                    if normalize_all:
                        dat_mat[ix_t, :] = tr / tr.max()
                    else:
                        dat_mat[ix_t, :] = tr
                    if plot_envelope:
                        dat_mat[ix_t, :] = envelope(dat_mat[ix_t, :])
                    if label_style == "month":
                        if UTCDateTime(t).strftime("%Y%m") not in months:
                            ylabels.append(t_to_plot_all[ix_t])
                            ylabelticks.append(UTCDateTime(t).strftime("%Y/%m/%d"))
                            months.append(UTCDateTime(t).strftime("%Y%m"))

                    elif label_style == "year":
                        if UTCDateTime(t).strftime("%Y") not in years:
                            ylabels.append(t_to_plot_all[ix_t])
                            ylabelticks.append(UTCDateTime(t).strftime("%Y/%m/%d"))
                            years.append(UTCDateTime(t).strftime("%Y"))
                   
                if plot_envelope:
                    vmin = 0
                    vmax = scale_factor_plotting * np.nanmax(dat_mat)
                else:
                    vmin = -scale_factor_plotting * np.nanmax(dat_mat)
                    vmax = scale_factor_plotting * np.nanmax(dat_mat)
                t_to_plot = t_to_plot_all
            ax1.pcolormesh(lag, t_to_plot, dat_mat, vmax=vmax, vmin=vmin,
                           cmap=cmap)

        if mark_17_quake:
            ylabels.append(UTCDateTime("2017,262").timestamp)
            ylabelticks.append("EQ Puebla")

        ax1.set_title(self.station_pair)
        ax1.set_ylabel("Normalized stacks (-)")
        ax1.set_xlim([seconds_to_start, seconds_to_show])
        ax1.set_xlabel("Lag (seconds)")
        ax1.set_yticks(ylabels)
        ax1.set_yticklabels(ylabelticks)
        ax1.yaxis.tick_right()

        ax1.grid(linestyle=":", color="lawngreen", axis="x")
        if seconds_to_show - seconds_to_start > 50:
            tickstep = 10.0
        elif seconds_to_show - seconds_to_start > 20:
            tickstep = 5.0
        elif seconds_to_show - seconds_to_start > 10:
            tickstep = 2.0
        else:
            tickstep = 1.0
        ax1.set_xticks([i for i in np.arange(seconds_to_start, seconds_to_show, tickstep)],
                       [str(i) for i in np.arange(seconds_to_start, seconds_to_show, tickstep)])

        if ax is None:
            if outfile is not None:
                plt.tight_layout()
                plt.savefig(outfile)
                plt.close()
            else:
                plt.show()
        else:
            return(ax1, t_to_plot)
