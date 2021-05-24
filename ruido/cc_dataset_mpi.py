import numpy as np
import h5py
from obspy import Trace, UTCDateTime
from obspy.signal.invsim import cosine_taper
import matplotlib.pyplot as plt
from scipy.signal import sosfilt, sosfiltfilt, hann, tukey, fftconvolve
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
from mpi4py import MPI
from warnings import warn
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


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
        ixs_nonzero = np.where(np.sum(self.data, axis=-1) != 0.0)[0]
        datanew = self.data[ixs_nonzero]
        tnew = self.timestamps[ixs_nonzero]
        if rank == 0:
            self.data = np.array(datanew)
            self.ntraces = self.data.shape[0]
            self.add_rms()
            self.median = np.nanmedian(self.data, axis=0)
            self.timestamps = np.array(tnew)
        else:
            self.ntraces = 0
            self.median = np.nan

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

    def align(self, t1, t2, ref, plot=False):
        l0 = np.argmin((self.lag - t1) ** 2)
        l1 = np.argmin((self.lag - t2) ** 2)
        
        taper = np.ones(self.lag.shape)
        taper[0: l0] = 0
        taper[l1:] = 0
        taper[l0: l1] = tukey(l1-l0)
        opt_shifts = np.zeros(self.timestamps.shape)
        for i in range(len(opt_shifts)):
            test = self.data[i] * taper
            ix0 = len(ref) // 2
            ix1 = len(ref) // 2 + len(ref)
            cc = fftconvolve(test[::-1] / test.max(), ref / ref.max(), "full")
            cc = cc[ix0: ix1]
            shift = int(self.lag[np.argmax(cc)] * self.fs)
            
            # apply the shift
            if shift == 0:
                pass
            elif shift > 0:
                self.data[i, shift: ] = self.data[i, : -shift].copy()
                self.data[i, 0: shift] = 0
            else:
                self.data[i, :shift ] = self.data[i, -shift:].copy()
                self.data[i, shift:] = 0


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
        self.dataset = {}

    def add_datafile(self, inputfile):

        self.datafile.close()
        self.station_pair = os.path.splitext(os.path.basename(inputfile))[0]
        self.station_pair = os.path.splitext(self.station_pair)[0]
        self.station_pair = os.path.splitext(self.station_pair)[0]
        self.datafile = h5py.File(inputfile, 'r')

    def __str__(self):
        if rank != 0:
            return(".")
        output = ""
        output += "Cross-correlation dataset: {}\n".format(self.station_pair)
        for (k, v) in self.dataset.items():
            output += "Contains {} traces on stacking level {}\n".format(v.ntraces, k)
            output += "Starting {}, ending {}\n".format(UTCDateTime(v.timestamps[0]).strftime("%d.%m.%Y"),
                                                        UTCDateTime(v.timestamps[-1]).strftime("%d.%m.%Y"))
        return(output)

    def data_to_envelope(self, stacklevel=1):
        #replace stacks by their envelope
        if rank != 0:
            raise ValueError("Call this function only on one process")
        newstacks = []
        for s in self.dataset[stacklevel].data:
            newstacks.append(envelope(s))
        self.dataset[stacklevel].data = np.array(newstacks)


    def data_to_memory(self, n_corr_max=None, keep_duration=0,
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
            raise NotImplementedError("use cc_dataset_newnewnew with old file format")

        if n_corr_max is None:
            n_corr_max = ntraces

        nshare = n_corr_max // size
        rest = n_corr_max % size

        # allocate data
        if rank == 0:
            try:
                alldata = np.zeros((n_corr_max, npts))
                alldatashare = np.zeros((n_corr_max - rest, npts))
                alltimestamps = np.zeros(n_corr_max)
                alltimestampsshare = np.zeros(n_corr_max - rest)
            except MemoryError:
                raise MemoryError("Data doesn't fit in memory, set a lower n_corr_max")
        else:
            alldata = None
            alldatashare = None
            alltimestampsshare = None

        partdata = np.zeros((nshare, npts))
        partdata[:, :] = self.datafile["corr_windows"]["data"][rank * nshare: rank * nshare + nshare, :]

        # allocate timestamps array
        timestamps = np.zeros(nshare)

        for i in range(nshare):
            tstamp = self.datafile["corr_windows"]["timestamps"][rank * nshare + i]
            tstmp = '{},{},{},{},{}'.format(*tstamp.split('.')[0: 5])
            timestamps[i] = UTCDateTime(tstmp).timestamp

        # gather
        comm.Gather(partdata, alldatashare, root=0)
        comm.Gather(timestamps, alltimestampsshare, root=0)

        if rank == 0:
            alltimestamps[0: n_corr_max - rest] = alltimestampsshare
            alldata[0: n_corr_max - rest] = alldatashare
            # get the rest!
            for ixdata in range(n_corr_max - rest, n_corr_max):
                alldata[ixdata] = self.datafile["corr_windows"]["data"][ixdata]
                tstamp = self.datafile["corr_windows"]["timestamps"][ixdata]
                tstmp = '{},{},{},{},{}'.format(*tstamp.split('.')[0: 5])
                alltimestamps[ixdata] = UTCDateTime(tstmp).timestamp
            print("Read to memory from {} to {}".format(UTCDateTime(alltimestamps[0]),
                                                        UTCDateTime(alltimestamps[-1])))
            if 0 in list(self.dataset.keys()):
                self.dataset[0].extend(alldata, alltimestamps, fs, keep_duration=keep_duration)
            else:
                self.dataset[0] = CCData(alldata, alltimestamps, fs)
        else:
            self.dataset = None

        # only debugging
        # if rank == 0:
        #     assert np.all(self.dataset[0].data[0:3] == self.datafile["corr_windows"]["data"][0:3])
        #     assert np.all(self.dataset[0].data[10:13] == self.datafile["corr_windows"]["data"][10:13])

    def select_by_percentile(self, ixs, stacklevel=0, perc=90, measurement="RMS",
                             mode="upper", debug_mode=False):
        # select on the basis of relative root mean square amplitude
        # of the cross-correlations
        if rank != 0:
            raise ValueError("Call this function only on one process")
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
        if rank != 0:
            raise ValueError("Call this function only on one process")

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
        if rank != 0:
            raise ValueError("Call this function only on one process")

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

        if rank != 0:
            raise ValueError("Call this function only on one process")

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

        try:  # elif stacklevel_out in list(self.dataset.keys()) and overwrite == False:
            self.dataset[stacklevel_out].extend([newstacks], [newt],
                                                self.dataset[stacklevel_in].fs, keep_duration=-1)
        except KeyError:
            self.dataset[stacklevel_out] = CCData([newstacks], [newt], self.dataset[stacklevel_in].fs)

    def data_to_dataframe(self, lag0, lag1, stacklevel, normalize=False):
        if rank != 0:
            raise ValueError("Call this function only on one process")
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

    def demean(self, stacklevel=0):
        if rank != 0:
            raise ValueError("Call this function only on one process")
        to_demean = self.dataset[stacklevel].data

        for d in to_demean:
            d -= d.mean()

    def detrend(self, stacklevel=0, order=3):
        if rank != 0:
            raise ValueError("Call this function only on one process")
        to_detrend = self.dataset[stacklevel].data
        for d in to_detrend:
            obspolynomial(d, order=order)

    def filter_data(self, stacklevel=0, taper_perc=0.1, filter_type="bandpass",
                    f_hp=None, f_lp=None, corners=4, zerophase=True,
                    maxorder=8, npool=1):

        if rank == 0:
            ndata = len(self.dataset[stacklevel].data)
            nshare = ndata // size
            nrest = ndata % size
            to_filter = self.dataset[stacklevel].data[0: ndata - nrest]
            npts = self.dataset[stacklevel].npts
            fs = self.dataset[stacklevel].fs
        else:
            nshare = None
            to_filter = None
            npts = None
            fs = None
            nrest = None
        fs = comm.bcast(fs, root=0)
        nshare = comm.bcast(nshare, root=0)
        nrest = comm.bcast(nrest, root=0)
        npts = comm.bcast(npts, root=0)
        to_filter_part = np.zeros((nshare, npts))

        # scatter the arrays
        comm.Scatter(to_filter, to_filter_part, root=0)

        # check that the input array has 2 dimensions
        if not np.ndim(to_filter_part) == 2:
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
            for i, tr in enumerate(to_filter_part):
                to_filter[i, :] = filter.cwt_bandpass(tr, f_hp, f_lp, df=fs)
        else:
            msg = 'Filter %s is not implemented, implemented filters:\
            bandpass, highpass,lowpass' % type
            raise ValueError(msg)

        if filter_type != "cwt":
            for i, tr in enumerate(to_filter_part):
                if zerophase:
                    to_filter_part[i, :] = sosfiltfilt(sos, taper * tr, padtype="even")
                else:
                    to_filter_part[i, :] = sosfilt(sos, taper * tr)

        # gather
        comm.Gather(to_filter_part, to_filter, root=0)

        # do the rest
        if rank == 0 and nrest > 0:
            filt_rest = []
            for ixdata in range(ndata - nrest, ndata):
                tr = self.dataset[stacklevel].data[ixdata, :]
                if zerophase:
                    filttr = sosfiltfilt(sos, taper * tr, padtype="even")
                else:
                    filttr = sosfilt(sos, taper * tr)
                filt_rest.append(filttr)
            self.dataset[stacklevel].data[ndata - nrest: ndata] = np.array(filt_rest)

    def get_window(self, t_mid, hw, lag, window_type, alpha=0.2):

        if rank != 0:
            raise ValueError("Call this function only on one process")
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
        if rank != 0:
            raise ValueError("Call this function only on one process")
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
        if rank != 0:
            raise ValueError("Call this function only on one process")
        ret = np.cumsum(a, dtype=np.complex)
        ret[n:] = ret[n:] - ret[:-n]
        return ret / n

    def window_data(self, stacklevel, t_mid, hw,
                    window_type="hann", tukey_alpha=0.2,
                    cutout=False):
        if rank != 0:
            raise ValueError("Call this function only on one process")
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

    def run_measurement(self, indices, to_measure, timestamps,
                        ref, fs, lag, f0, f1, ngrid,
                        dvv_bound, method="stretching"):
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


        #cnt = 0
        for i, tr in enumerate(to_measure):
            if i not in indices:
                dvv[cnt, :] = np.nan
                dvv_times[cnt] = timestamps[i]
                ccoeff[cnt] = np.nan
                # print(ccoeff[cnt])
                best_ccoeff[cnt] = np.nan
                dvv_error[cnt, :] = np.nan
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

            dvv[i, :] = dvvp
            dvv_times[i] = timestamps[i]
            ccoeff[i] = cdpp
            # print(ccoeff[cnt])
            best_ccoeff[i] = coeffp
            dvv_error[i, :] = delta_dvvp
            # cnt += 1
        return(dvv, dvv_times, ccoeff, best_ccoeff, dvv_error)

    def measure_dvv_par(self, ref, f0, f1, stacklevel=1, method="stretching",
                        ngrid=100, dvv_bound=0.03,
                        measure_smoothed=False, indices=None,
                        moving_window_length=None, slide_step=None, maxlag_dtw=0.0,
                        len_dtw_msr=None):
        # "WTF! Why is this so complicated??" -- I handled it this way because only rank 0 actually has the data
        # The reason for this is that if we create copies of the entire dataset on multiple ranks, 
        # the memory usage will drastically increase
        # in the future maybe re-write it all so that each rank holds a chunk of the data (maybe better 
        # but I don't have time to put that in rn). Also more complicated as ranks will have to exchange specific data.
        if indices is not None:
            if len(indices) < 10:
                warn("This method measures all windows, if you are only planning to measure a few, run measure_dvv on one process.")
        if rank == 0:
            ndata = len(self.dataset[stacklevel].data)
            nshare = ndata // size
            nrest = ndata % size

            to_measure = self.dataset[stacklevel].data[0: ndata - nrest]
            timestamps = self.dataset[stacklevel].timestamps[0: ndata - nrest]
            npts = self.dataset[stacklevel].npts
            fs = self.dataset[stacklevel].fs
            lag = self.dataset[stacklevel].lag
        else:
            ndata = None
            nshare = None
            to_measure = None
            timestamps = None
            npts = None
            fs = None
            nrest = None
            lag = None
        fs = comm.bcast(fs, root=0)
        nshare = comm.bcast(nshare, root=0)
        ndata = comm.bcast(ndata, root=0)
        nrest = comm.bcast(nrest, root=0)
        npts = comm.bcast(npts, root=0)
        lag = comm.bcast(lag, root=0)

        to_measure_part = np.zeros((nshare, npts))
        timestamps_part = np.zeros(nshare)
        dvv_all = np.zeros((ndata, 1))
        dvv_error_all = np.zeros((ndata, 1))
        timestamps_all = np.zeros((ndata, 1))
        ccoeff_all = np.zeros((ndata, 1))
        best_ccoeff_all = np.zeros((ndata, 1))
        # scatter the arrays
        comm.Scatter(to_measure, to_measure_part, root=0)
        comm.Scatter(timestamps, timestamps_part, root=0)

        # print("rank {}, nr traces to measure {}".format(rank, len(to_measure_part)))

        dvv, dvv_times, ccoeff, best_ccoeff, dvv_error = \
        self.run_measurement(indices, to_measure_part, timestamps_part,
                             ref, fs, lag, f0, f1, ngrid,
                             dvv_bound, method="stretching")

        comm.Gather(dvv, dvv_all[0: ndata - nrest], root=0)
        comm.Gather(dvv_times, timestamps_all[: ndata - nrest], root=0)
        comm.Gather(dvv_error, dvv_error_all[: ndata - nrest], root=0)
        comm.Gather(ccoeff, ccoeff_all[: ndata - nrest], root=0)
        comm.Gather(best_ccoeff, best_ccoeff_all[: ndata - nrest], root=0)

        if rank == 0:
            print(dvv_all.shape)
            print(ndata)
            print(nrest)
            if nrest > 0:
                to_measure_extra = self.dataset[stacklevel].data[ndata - nrest:]
                timestamps_extra = self.dataset[stacklevel].timestamps[ndata - nrest:]
                if indices is None:
                    indices = range(len(to_measure_extra))
                dvv, dvv_times, ccoeff, best_ccoeff, dvv_error = \
                self.run_measurement(indices, to_measure_extra, timestamps_extra,
                                     ref, fs, lag, f0, f1, ngrid,
                                     dvv_bound, method="stretching")
              
                dvv_all[ndata - nrest:, :] = dvv
                timestamps_all[ndata - nrest:, 0] = dvv_times
                ccoeff_all[ndata - nrest:, 0] = ccoeff
                best_ccoeff_all[ndata - nrest:, 0] = best_ccoeff
                dvv_error_all[ndata - nrest:, :] = dvv_error
            else:
                pass
        else:
            pass

        comm.barrier()

        if rank == 0:
            return(dvv_all, timestamps, ccoeff_all, best_ccoeff_all, dvv_error_all, [])
        else:
            return([],[],[],[],[],[])


    def interpolate_stacks(self, new_fs, stacklevel=1):
        if rank != 0:
            raise ValueError("Call this function only on one process")
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
        if rank != 0:
            raise ValueError("Call this function only on one process")
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
        if rank != 0:
            raise ValueError("Call this function only on one process")
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

    def measure_dvv(self, ref, f0, f1, stacklevel=1, method="stretching",
                    ngrid=90, dvv_bound=0.03,
                    measure_smoothed=False, indices=None,
                    moving_window_length=None, slide_step=None, maxlag_dtw=0.0,
                    len_dtw_msr=None):

        if rank == 0:
            cwtfreqs = None
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
            ccoeff = np.zeros((len(indices), 1))
            best_ccoeff = np.zeros((len(indices), 1))

            dvv = np.zeros((len(indices), 1))
            dvv_error = np.zeros((len(indices), 1)) 

            cnt = 0
            for i, tr in enumerate(to_measure):
                if i not in indices:
                    continue

                if method == "stretching":
                    dvvp, delta_dvvp, coeffp, cdpp = stretching_vect(reference, tr,
                                                            dvv_bound, ngrid, para)
                    cwtfreqs = []
                else:
                    raise NotImplementedError("I have only stretching here.")
                dvv[cnt, :] = dvvp
                dvv_times[cnt] = timestamps[i]
                ccoeff[cnt, :] = cdpp
                # print(ccoeff[cnt])
                best_ccoeff[cnt, :] = coeffp
                dvv_error[cnt, :] = delta_dvvp
                cnt += 1
            return(dvv, dvv_times, ccoeff, best_ccoeff, dvv_error, cwtfreqs)
        else:
            return([],[],[],[],[], [])
