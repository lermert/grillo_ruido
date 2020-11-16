import numpy as np
import h5py
from obspy import Trace, UTCDateTime
from obspy.signal.invsim import cosine_taper
import matplotlib.pyplot as plt
from scipy.signal import sosfilt, sosfiltfilt, hann, tukey
from scipy.interpolate import interp1d
from ruido.utils import filter
import pandas as pd
import os
from noisepy.noise_module import stretching, dtw_dvv, stretching_vect, wts_dvv
from ruido.clustering import cluster
from obspy.signal.filter import envelope
from obspy.signal.detrend import polynomial as obspolynomial

hour_strings = {"": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                "eight": 8, "ten": 10, "twelve": 12, "twenty": 20}



class CCDataset(object):
    """Processing object for Cross-correlation dataset organized by time stamp in hdf5 file."""

    def __init__(self, inputfile, ref=None, exclude_tag=None):
        """
        :type inputfile: string
        :param inputfile: File path to hdf5 file
        :type refs: numpy.ndarray
        :param refs: reference trace, e.g. from another file

        """
        super(CCDataset, self).__init__()
        self.station_pair = os.path.splitext(os.path.basename(inputfile))[0]
        self.station_pair = os.path.splitext(self.station_pair)[0]
        self.station_pair = os.path.splitext(self.station_pair)[0]
        self.datafile = h5py.File(inputfile, 'r')

        self.initialize_dataset(inputfile)
        
        self.stacks = {}

        

    def initialize_dataset(self, inputfile):
        self.station_pair = os.path.splitext(os.path.basename(inputfile))[0]
        self.station_pair = os.path.splitext(self.station_pair)[0]
        self.station_pair = os.path.splitext(self.station_pair)[0]
        self.datafile = h5py.File(inputfile, 'r')
        self.fs = dict(self.datafile['stats'].attrs)['sampling_rate']
        self.delta = 1. / self.fs

        if "data" in list(self.datafile["corr_windows"].keys()):
            self.npts = self.datafile['corr_windows']["data"][0].shape[0]
            self.ntraces = len(np.where(self.datafile["corr_windows"]["timestamps"][:] != "")[0])
        else:  #  old file format
            self.npts = self.datafile['corr_windows'][next(iter(self.datafile["corr_windows"]))].shape[0]
            self.ntraces = len(self.datafile["corr_windows"])
        self.max_lag = (self.npts - 1) / 2 / self.fs
        self.lag = np.linspace(-self.max_lag, self.max_lag, self.npts)

        self.data = None
        self.datakeys = []
        self.timestamps = []
        self.ref = np.zeros(self.npts)
        self.rms = None
        self.median = None


    def __str__(self):
        output = ""
        output += "Cross-correlation dataset: {}\n".format(self.station_pair)
        output += "Maximum lag: {} seconds\n".format(self.max_lag)
        output += "Sampling rate: {} Hz\n".format(self.fs)
        output += "Number of traces: {}\n".format(self.ntraces)
        try:
            output += "Start date: {}\n".format((self.datafile["corr_windows"]["timestamps"][0]))
            i = -1
            enddate = ""
            while enddate == "":
                i -= 1
                enddate = self.datafile["corr_windows"]["timestamps"][i]
            output += "End date: {}\n".format(enddate)
        except KeyError:
            pass
        return(output)


    def data_to_memory(self, n_corr_max=None, t_min=None, t_max=None):

        if n_corr_max is None:
            n_corr_max = self.ntraces

        dat_shape = int(n_corr_max)
        try:
            self.data = np.zeros((dat_shape, self.npts))
        except MemoryError:
            print("Data doesn't fit in memory, try setting a lower n_corr_max\
or a higher prestack")
            return()

        self.datakeys = np.zeros(dat_shape, dtype=np.str)
        self.timestamps = np.zeros(dat_shape)
        try:  # new file format
            for i, v in enumerate(self.datafile["corr_windows"]["data"][:]):
                if i == n_corr_max:
                    break

                self.data[i, :] = v[:]
                tstamp = self.datafile["corr_windows"]["timestamps"][i]
                self.datakeys[i] = tstamp

                tstmp = '{},{},{},{},{}'.format(*tstamp.split('.')[0: 5])
                self.timestamps[ix] = UTCDateTime(tstmp).timestamp

        except KeyError:  # old file format
            for i, (k, v) in enumerate(self.datafile["corr_windows"].items()):
                if i == n_corr_max:
                    break
                self.data[i, :] = v[:]
                self.datakeys[i] = k
                tstmp = '{},{},{},{},{}'.format(*k.split('.')[0: 5])
                self.timestamps[i] = UTCDateTime(tstmp).timestamp


        if t_min is not None:
            ix0 = np.argmin(abs(self.timestamps - t_min))
        else:
            ix0 = 0
        if t_max is not None:
            ix1 = np.argmin(abs(self.timestamps - t_max))
        else:
            ix1 = self.ntraces
        self.ncorr = 
        self.data = self.data[ix0: ix1]
        self.timestamps = self.timestamps[ix0: ix1]
        self.datakeys = self.datakeys[ix0: ix1]
        self.ntraces = ix1 - ix0
        print("Read to memory from {} to {}".format(UTCDateTime(self.timestamps[0]),
                                                    UTCDateTime(self.timestamps[-1])))

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


    def add_rms(self):

        rms = np.zeros(self.data.shape[0])

        if len(rms) == 0:
            return

        for i, dat in enumerate(self.data):
            rms[i] = np.sqrt(np.sum((dat - dat.mean()) ** 2) / len(dat))
           
        self.rms = rms

    def select_by_percentile(self, ixs, perc, measurement="RMS",
                             mode="upper", debug_mode=False):

        if self.rms is None:
            self.add_rms()
        rms = self.rms[ixs]

        if mode == "upper":
            ixs_keep = np.where(rms <= np.percentile(rms, perc))
        elif mode == "lower":
            ixs_keep = np.where(rms >= np.percentile(rms, perc))
        elif mode == "both":
            ixs_keep = np.intersect1d(np.where(rms <= np.percentile(rms, perc)),
                                      np.where(rms >= np.percentile(rms, 100 - perc)))
        if debug_mode:
            print("Selection by percentile of RMS: Before, After", len(ixs), len(ixs_keep))

        return(ixs[ixs_keep])


    def group_for_stacking(self, t0, duration, kmeans_label=None):
        """
        Create a list of time stamps to be stacked
        --> figure out what "belongs together" in terms of time window
        afterwards, more elaborate selections can be applied :)
        """

        # find closest to t0 window
        assert type(t0) == float, "t0 must be floating point time stamp"
        ix_start = np.argmin((self.timestamps - t0) ** 2)

        # find indices
        ixs_selected = np.intersect1d(np.where(self.timestamps >= t0),
                                      np.where(self.timestamps < (t0 + duration)))

        # check if anything to do for clusters
        if kmeans_label is not None:
            ixs_selected = np.intersect1d(ixs_selected, np.where(self.kmeans_labels == kmeans_label))

        return(ixs_selected)

    def select_for_stacking(self, ixs, selection_mode, bootstrap=False, cc=0.5,
                            twin=None, **kwargs):
        """
        Select by: closeness to reference or percentile or...
        """

        if len(ixs) == 0:
            return([])

        if selection_mode == "rms_percentile":
            ixs_selected = self.select_by_percentile(ixs, **kwargs)
        elif selection_mode == "cc_to_median":
            if self.median is None:
                self.median = np.nanmedian(self.data, axis=0)
                # print(self.median.shape)
            ixs_selected = []
            if twin is not None:
                cc_ixs = [np.argmin((self.lag - t) ** 2) for t in twin]
            else:
                cc_ixs = [0, -1]
            for i in ixs:
                corrcoeff = np.corrcoef(self.data[i, cc_ixs[0]: cc_ixs[1]], 
                                        self.median[cc_ixs[0]: cc_ixs[1]])[0, 1]
                if corrcoeff < cc or np.isnan(corrcoeff):
                    continue
                ixs_selected.append(i)
            ixs_selected = np.array(ixs_selected)
        elif selection_mode == "cc_to_ref":
            ixs_selected = []
            if twin is not None:
                cc_ixs = [np.argmin((self.lag - t) ** 2) for t in twin]
            else:
                cc_ixs = [0, -1]
            for i in ixs:
                corrcoeff = np.corrcoef(self.data[i, cc_ixs[0]: cc_ixs[1]],
                                        self.ref[cc_ixs[0]: cc_ixs[1]])[0, 1]
                if corrcoeff < cc or np.isnan(corrcoeff):
                    continue
                ixs_selected.append(i)
            ixs_selected = np.array(ixs_selected)

        else:
            raise NotImplementedError

        if bootstrap:
            ixs_selected = np.random.choice(ixs_selected, len(ixs_selected))  # sampling with replacing

        return(ixs_selected)

    def stack(self, ixs, mode="linear"):

        if mode == "linear":
            s = self.data[ixs].sum(axis=0)
            s = Trace(s / len(ixs))
            s.stats.sampling_rate = self.fs
            self.stacks[self.timestamps[ixs[0]]] = s


    def data_to_dataframe(self, lag0, lag1, normalize=False):

        ixs_get0 = np.where(self.lag >= lag0)
        ixs_lets = np.where(self.lag <= lag1)
        ixs = np.intersect1d(ixs_get0, ixs_lets)
        self.df_lags = self.lag[ixs]
        nr_samples = len(ixs)

        self.df = pd.DataFrame(columns=range(nr_samples),
                               data=np.zeros((self.ncorr, nr_samples)))
        for i in range(self.ncorr):
            for j in range(nr_samples):
                if normalize:
                    self.df.iat[i, j] = self.data[i, ixs][j] / (self.data[i, :].max() + np.finfo(float).eps)
                else:
                    self.df.iat[i, j] = self.data[i, ixs][j]

    def stacks_to_dataframe(self, lag0, lag1, normalize=True):

        tstamps = np.array(list(self.stacks.keys()))
        ixs_get0 = np.where(self.lag >= lag0)
        ixs_lets = np.where(self.lag <= lag1)
        ixs = np.intersect1d(ixs_get0, ixs_lets)
        self.df_lags = self.lag[ixs]
        nr_samples = len(ixs)

        self.df = pd.DataFrame(columns=range(nr_samples),
                               data=np.zeros((len(self.stacks), nr_samples)))
        for i in range(len(self.stacks)):
            for j in range(nr_samples):
                if normalize:
                    self.df.iat[i, j] = self.data[i, ixs][j] / (self.data[i, :].max() + np.finfo(float).eps)
                else:
                    self.df.iat[i, j] = self.data[i, ixs][j]


    def data_from_mem(self, ix=None, k=None):
        if ix is None:
            return self.data[self.datakeys.index(k), :]
        else:
            return self.data[ix, :]

    def reference(self, reftype, t_min=None, t_max=None, overwrite=False):

        """
        t_min and t_max are start- and endtimes for forming a reference
        If None are given, then all windows are used.
        """

        if self.ref.sum() != 0.0 and not overwrite:
            raise ValueError("Reference was already given, not overwriting.")

        if False in [t_min is None, t_max is None]:
            if t_min is not None and t_max is None:
                ixs = np.where(np.array(self.timestamps) >= t_min)[0]
            elif t_min is None and t_max is not None:
                ixs = np.where(np.array(self.timestamps) <= t_max)[0]
            else:
                ixs = np.where(np.array(self.timestamps) >= t_min)[0]  
                ixs = np.where(np.array(self.timestamps)[ixs] <= t_max)[0]
        else:
            ixs = np.where(np.array(self.timestamps) > 0)[0]  # everywhere
        

        if reftype == "arithmetic_mean":
            if self.data is not None:
                self.ref = np.mean(self.data[ixs], axis=0)

        elif reftype == "median":
            if self.data is not None:
                self.ref = np.nanmedian(self.data[ixs], axis=0)


    def filter_stacks(self, taper_perc=0.05, filter_type="bandpass",
                      f_hp=None, f_lp=None, corners=4, zerophase=True,
                      maxorder=8):
        if filter_type == "bandpass":
            for (k, tr) in self.stacks.items():
                tr.taper(taper_perc)
                tr.filter(filter_type, freqmin=f_hp,
                          freqmax=f_lp, corners=corners, zerophase=zerophase)
        elif filter_type == "cheby2_bandpass":
            sos = filter.cheby2_bandpass(df=self.fs, freq0=f_hp, freq1=f_lp,
                                         maxorder=maxorder)

            taper = cosine_taper(self.npts, taper_perc)
            for (k, tr) in self.stacks.items():
                if zerophase:
                    firstpass = sosfilt(sos, taper * tr.data)
                    tr.data = sosfilt(sos, firstpass[::-1])[::-1]
                    # then assign to newfile, which might be self.file
                else:
                    tr.data = sosfilt(sos, taper * tr.data)
        elif filter_type == "cwt":
            taper = cosine_taper(self.npts, taper_perc)
            for (k, tr) in self.stacks.items():
                tr.data = filter.cwt_bandpass(taper * tr.data, f_hp, f_lp, df=self.fs)

    def demean(self):

        for d in self.data:
            d -= d.mean()

    def detrend(self, order=3):
        for d in self.data:
            obspolynomial(d, order=order)

    # To do: common filter function for data, stacks reference!!!
    def filter_reference(self, taper_perc=0.05, filter_type="bandpass",
                         f_hp=None, f_lp=None, corners=4, zerophase=True,
                         maxorder=8):
        taper = cosine_taper(self.npts, taper_perc)
        if filter_type == 'bandpass':
            if None in [f_hp, f_lp]:
                raise TypeError("f_hp and f_lp (highpass and lowpass frequency) must be floats.")
            sos = filter.bandpass(df=self.fs, freqmin=f_hp, freqmax=f_lp,
                                  corners=corners)
        elif filter_type == 'lowpass':
            sos = filter.lowpass(df=self.fs, freq=f_lp, corners=corners)
        elif filter_type == 'highpass':
            sos = filter.highpass(df=self.fs, freq=f_hp, corners=corners)
        elif filter_type == "cheby2_bandpass":
            sos = filter.cheby2_bandpass(df=self.fs, freq0=f_hp, freq1=f_lp,
                                         maxorder=maxorder)
        elif filter_type == "cwt":
            taper = cosine_taper(self.npts, taper_perc)
            self.ref = filter.cwt_bandpass(taper * self.ref, f_hp, f_lp, df=self.fs)
        else:
            msg = 'Filter %s is not implemented, implemented filters:\
            bandpass, highpass,lowpass' % type
            raise ValueError(msg)

        if filter_type != "cwt":
            if zerophase:
                firstpass = sosfilt(sos, taper * self.ref)
                self.ref = sosfilt(sos, firstpass[::-1])[::-1]
                # then assign to newfile, which might be self.file
            else:
                self.ref = sosfilt(sos, taper * self.ref)

    def filter_data(self, taper_perc=0.1, filter_type="bandpass",
                    f_hp=None, f_lp=None, corners=4, zerophase=True,
                    maxorder=8):

        taper = cosine_taper(self.npts, taper_perc)
        if filter_type == 'bandpass':
            if None in [f_hp, f_lp]:
                raise TypeError("f_hp and f_lp (highpass and lowpass frequency) must be floats.")
            sos = filter.bandpass(df=self.fs, freqmin=f_hp, freqmax=f_lp,
                                  corners=corners)
        elif filter_type == 'lowpass':
            sos = filter.lowpass(df=self.fs, freq=f_lp, corners=corners)
        elif filter_type == 'highpass':
            sos = filter.highpass(df=self.fs, freq=f_hp, corners=corners)
        elif filter_type == "cheby2_bandpass":
            sos = filter.cheby2_bandpass(df=self.fs, freq0=f_hp, freq1=f_lp,
                                         maxorder=maxorder)
        elif filter_type == "cwt":
            taper = cosine_taper(self.npts, taper_perc)
            for i, tr in enumerate(self.data):
                self.data[i, :] = filter.cwt_bandpass(tr, f_hp, f_lp, df=self.fs)
        else:
            msg = 'Filter %s is not implemented, implemented filters:\
            bandpass, highpass,lowpass' % type
            raise ValueError(msg)

        if filter_type != "cwt":
            for i, tr in enumerate(self.data):

                if zerophase:
                    tr *= taper
                    self.data[i, :] = sosfiltfilt(sos, tr, padtype="even")
                    #firstpass = sosfilt(sos, taper * tr)
                    #self.data[i, :] = sosfilt(sos, firstpass[::-1])[::-1]
                    # then assign to newfile, which might be self.file
                else:
                    self.data[i, :] = sosfilt(sos, taper * tr)


    def get_window(self, t_mid, hw, window_type, alpha=0.2):

        # half_lag_minus_one = int((self.npts - 1) / 2)
        ix_1 = np.argmin(abs(self.lag - t_mid - hw))
        ix_0 = np.argmin(abs(self.lag - t_mid + hw))  # [self.lag > 0]
        win = np.zeros(self.lag.shape)
        if window_type == "hann":
            # win[half_lag_minus_one + ix_0 : half_lag_minus_one + ix_1] = hann(ix_1 - ix_0)
            win[ix_0: ix_1] = hann(ix_1 - ix_0)
        elif window_type == "tukey":
            win[ix_0: ix_1] = tukey(ix_1 - ix_0, alpha=alpha)
        elif window_type == "boxcar":
            win[ix_0: ix_1] = 1.0
        return(win)

    def window_stacks(self, t_mid, hw, window_type="hann", tukey_alpha=0.2, overwrite=False):

        win = self.get_window(t_mid, hw, window_type=window_type, alpha=tukey_alpha)

        if overwrite:
            for (k, stack) in self.stacks.items():
                self.stacks[k].data *= win
        else:
            self.windowed_stacks = {}
            for (k, stack) in self.stacks.items():
                self.windowed_stacks[k] = Trace((stack.data * win).copy())
                self.windowed_stacks[k].stats.sampling_rate = stack.stats.sampling_rate

    def window_reference(self, t_mid, hw, window_type="hann", tukey_alpha=0.2):

        win = self.get_window(t_mid, hw, window_type=window_type, alpha=tukey_alpha)
        return(self.ref * win)

    def window_data(self, t_mid, hw, window_type="hann", tukey_alpha=0.2):
        win = self.get_window(t_mid, hw, window_type=window_type, alpha=tukey_alpha)
        for ix in range(self.data.shape[0]):
            self.data[ix, :] *= win
            self.data[ix, :] = self.data[ix, :]


    def measure_dvv(self, f0, f1, method="stretching", ngrid=90, dvv_bound=0.03,
                    do_filter_reference=False, measure_smoothed=False, indices=None):

        if len(self.stacks) == 0:
            return()

        reference = self.ref.copy()
        para = {}
        para["dt"] = 1. / self.fs
        para["twin"] = [self.lag[0], self.lag[-1] + 1. / self.fs]  # [self.lag[0], self.lag[-1] + 1. / self.fs]
        para["freq"] = [f0, f1]


        if do_filter_reference:
            sos = filter.cheby2_bandpass(df=self.fs, freq0=f0, freq1=f1,
                                         maxorder=12)
            taper = cosine_taper(self.npts, 0.2)
            firstpass = sosfilt(sos, taper * reference)
            reference = sosfilt(sos, firstpass[::-1])[::-1]

        if indices is None:
            indices = range(len(self.stacks))

        self.dvv_times = np.zeros(len(indices))
        self.ccoeff = np.zeros(len(indices))
        self.best_ccoeff = np.zeros(len(indices))
       

        if method in ["stretching"]:
            self.dvv = np.zeros(len(indices))
            self.dvv_error = np.zeros(len(indices))
        elif method in ["cwt-stretching"]:
            testmsr = wts_dvv(reference, reference, True, 
                              para, dvv_bound, ngrid)
            self.dvv = np.zeros((len(indices), len(testmsr[1])))
            self.dvv_error = np.zeros((len(indices), len(testmsr[1])))

        else:
            raise ValueError("Unknown measurement method {}.".format(method))


        cnt = 0
        for i, (k, tr) in enumerate(list(self.stacks.items())):
            if i not in indices:
                continue
            if method == "stretching":
                dvv, delta_dvv, coeff, cdp = stretching_vect(reference, tr.data,
                                                        dvv_bound, ngrid, para)
            elif method == "dtw":
                dvv, err, dist = dtw_dvv(reference, tr.data,
                                         para, maxLag=int(1. * self.fs),
                                         b=10, direction=1)
                coeff = err
                cdp = np.nan
                delta_dvv = np.nan
            elif method == "cwt-stretching":
                cwtfreqs, dvv, err = wts_dvv(reference, tr.data, True, 
                                             para, dvv_bound, ngrid)
                delta_dvv = err
                coeff = np.nan
                cdp = np.nan
                self.cwt_freqs = cwtfreqs

            self.dvv[cnt, :] = dvv
            self.dvv_times[cnt] = k
            self.ccoeff[cnt] = cdp
            self.best_ccoeff[cnt] = coeff
            self.dvv_error[cnt] = delta_dvv
            cnt += 1
    # def interpolate_reference(self, new_fs):
    #     new_npts = int(len(self.ref) / self.fs * new_fs)
    #     new_npts = int(len(self.ref) / self.fs * new_fs)
    #     if new_npts % 2 == 0:
    #         new_npts += 1

    #     new_lag = np.linspace(-self.max_lag, self.max_lag, new_npts)
    #     if len(new_lag) > len(self.ref):
    #         print("old lag")
    #         old_lag = np.linspace(self.lag[0], self.lag[-1], len(self.ref))
    #         f = interp1d(old_lag, self.ref, kind="cubic")
    #     else:
    #         f = interp1d(self.lag, self.ref, kind="cubic")
    #         self.lag = new_lag
    #         self.fs = new_fs
    #     self.ref = f(new_lag)

    def interpolate_stacks(self, new_fs):

        if (new_fs % self.fs) != 0:
            raise ValueError("For the moment only integer-factor resampling is permitted.")

        new_npts = int((self.npts - 1.) / self.fs * new_fs) + 1
        print("new_npts: ", new_npts)

        new_lag = np.linspace(-self.max_lag, self.max_lag, new_npts)
        for (k, stack) in self.stacks.items():
            f = interp1d(self.lag, stack.data, kind="cubic")
            stack.data = f(new_lag)
        f = interp1d(self.lag, self.ref, kind="cubic")
        self.ref = f(new_lag)
        self.lag = new_lag
        self.npts = new_npts
        self.fs = (self.npts - 1) / (self.lag[-1] - self.lag[0])
        self.delta = 1. / self.fs
        print(self.delta, self.fs, self.lag.max())
        self.data = None

    def form_clusters(self, n_clusters, mode="stacks",
                      lag0=0, lag1=20, n_corr_max=None,
                      normalize=False, method="kmeans",
                      n_iterations=100):

        if mode == "stacks":
            # stacks are used for clustering
            self.stacks_to_dataframe(lag0=lag0, lag1=lag1, normalize=normalize)
        else:
            self.data_to_dataframe(lag0=lag0, lag1=lag1, normalize=normalize)

        if method == "kmeans":
            self.kmeans_labels, self.centroids, self.inertia, self.n_it_cluster = cluster(self.df, n_clusters)
        elif method == "minibatch":
            self.kmeans_labels, self.centroids, self.inertia, self.n_it_cluster = cluster(self.df, n_clusters, n_iterations)

    def plot_stacks(self, outfile=None, seconds_to_show=20, scale_factor_plotting=0.1,
        plot_mode="heatmap", seconds_to_start=0.0, cmap=plt.cm.bone, mask_gaps=False, step=None, figsize=None,
        color_by_cc=False, normalize_all=False, first_of_month_label=False, ax=None, plot_envelope=False, yearlabel=True):

        if mask_gaps and step == None:
            raise ValueError("To mask the gaps, you must provide the step between successive windows.")

        to_plot = self.stacks
        if len(to_plot) == 0:
            return()
        ylabels = []
        ylabelticks = []

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax1 = fig.add_subplot(111)
        else:
            ax1 = ax

        if plot_mode == "traces":            
            cnt = 0
            for (tk, t) in self.stacks.items():

                if color_by_cc:
                    cmap = plt.cm.get_cmap("Spectral", 20)
                    crange = np.linspace(-1.0, 1.0, 20)
                    cmap = cmap(crange)
                    cccol = np.corrcoef(t.data, self.ref)[0][1]
                    ix_colr = np.argmin((cccol - crange) ** 2)
                    ax1.plot(self.lag, t.data / t.data.max() + scale_factor_plotting * cnt,
                             c=cmap[ix_colr], alpha=0.5)
                else:
                    ax1.plot(self.lag, t.data / t.data.max() + scale_factor_plotting * cnt,
                             'k', alpha=0.5, linewidth=0.5)
                cnt += 1
                if first_of_month_label:
                    if UTCDateTime(tk).strftime("%d") == "01"\
                       and UTCDateTime(tk).strftime("%d.%m.%y") not in ylabelticks:
                        ylabels.append(scale_factor_plotting * cnt)
                        ylabelticks.append(UTCDateTime(tk).strftime("%d.%m.%y"))
                else:
                    if cnt == 0:
                        ylabelticks.append(UTCDateTime(tk).strftime("%d.%m.%y"))
                        ylabels.append(scale_factor_plotting * cnt)
                    elif cnt - ylabels[-1] / scale_factor_plotting == len(to_plot) // 6:
                        ylabelticks.append(UTCDateTime(tk).strftime("%d.%m.%y"))
                        ylabels.append(scale_factor_plotting * cnt)
            if self.ref.sum() != 0:
                ax1.plot(self.lag, self.ref / self.ref.max(), linewidth = 2.0)
                ax1.set_ylim([-1, scale_factor_plotting * cnt + np.mean(abs(t.data))])
            else:
                ax1.set_ylim([0 - np.mean(abs(t.data)), 
                             scale_factor_plotting * cnt + np.mean(abs(t.data))])

        elif plot_mode == "heatmap":

            if not mask_gaps:
                dat_mat = np.zeros((len(to_plot), self.npts))
                t_to_plot = np.zeros(len(to_plot))
                for ix, (k, t) in enumerate(to_plot.items()):
                    if normalize_all:
                        dat_mat[ix, :] = t.data / t.data.max()
                    else:
                        dat_mat[ix, :] = t.data
                    if plot_envelope:
                        dat_mat[ix, :] = envelope(dat_mat[ix, :])
                    t_to_plot[ix] = k
                    if first_of_month_label:
                        if UTCDateTime(k).strftime("%d") == "01"\
                            and UTCDateTime(k).strftime("%d.%m.%y") not in ylabelticks:
                            ylabels.append(ix)
                            ylabelticks.append(UTCDateTime(k).strftime("%d.%m.%y"))
                    else:
                        if ix == 0:
                            ylabels.append(ix)
                            ylabelticks.append(UTCDateTime(k).strftime("%d.%m.%y"))
                        elif ix - ylabels[-1] == len(to_plot) // 6:
                            ylabels.append(ix)
                            ylabelticks.append(UTCDateTime(k).strftime("%d.%m.%y"))
                if plot_envelope:
                    vmin = 0
                    vmax = scale_factor_plotting*dat_mat.max()
                else:
                    vmin = -scale_factor_plotting * dat_mat.max()
                    vmax = scale_factor_plotting * dat_mat.max() 
                ax1.pcolormesh(self.lag, range(len(to_plot)), dat_mat,
                               vmax=vmax,
                               vmin=vmin,
                               cmap=cmap)
            else:
                tstamp0 = list(to_plot.keys())[0]
                tstamp1 = list(to_plot.keys())[-1]
                #print(UTCDateTime(tstamp0), UTCDateTime(tstamp1))
                t_to_plot = np.arange(tstamp0, tstamp1 + step, step=step)
                dat_mat = np.zeros((len(t_to_plot), self.npts))
                dat_mat[:, :] = np.nan
                years = []
                for ix, (k, t) in enumerate(to_plot.items()):

                    ix_t = np.argmin(np.abs(t_to_plot - k))
                    if normalize_all:
                        dat_mat[ix_t, :] = t.data / t.data.max()
                    else:
                        dat_mat[ix_t, :] = t.data
                    if plot_envelope:
                        dat_mat[ix_t, :] = envelope(dat_mat[ix, :])
                    
                    if first_of_month_label:
                        if UTCDateTime(k).strftime("%d") == "01"\
                            and UTCDateTime(k).strftime("%d.%m.%y") not in ylabelticks:
                            ylabels.append(t_to_plot[ix_t])
                            ylabelticks.append(UTCDateTime(k).strftime("%Y/%m"))

                    elif yearlabel:
                       
                        if UTCDateTime(k).strftime("%Y") not in years:
                            years.append(UTCDateTime(k).strftime("%Y"))
                            ylabels.append(t_to_plot[ix_t])
                            ylabelticks.append(UTCDateTime(k).strftime("%Y/%m"))
                    else:
                        if ix_t == 0:
                            ylabels.append(t_to_plot[ix_t])
                            ylabelticks.append(UTCDateTime(k).strftime("%Y/%m"))
                        elif ix_t % (int(len(t_to_plot) // 6)) == 0:
                            ylabels.append(t_to_plot[ix_t])
                            ylabelticks.append(UTCDateTime(k).strftime("%Y/%m"))
                if plot_envelope:
                    vmin = 0
                    vmax = scale_factor_plotting * np.nanmax(dat_mat)
                else:
                    vmin = -scale_factor_plotting * np.nanmax(dat_mat)
                    vmax = scale_factor_plotting * np.nanmax(dat_mat)
                ax1.pcolormesh(self.lag, t_to_plot, dat_mat,
                               vmax=vmax,
                               vmin=vmin,
                               cmap=cmap)

        #ylabels.append(UTCDateTime("2017,262").timestamp)
        #ylabelticks.append("Puebla")

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
