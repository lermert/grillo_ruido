import numpy as np
import h5py
from obspy import Trace, UTCDateTime
import matplotlib.pyplot as plt
from scipy.signal import sosfilt, hann
from scipy.interpolate import interp1d
#from ruido.utils import filter
from utils import filter
import pandas as pd
import os
from clustering import cluster

hour_strings = {"three": 3.0, "four": 4.0, "six": 6.0, "eight": 8.0, "twelve": 12.}

class CCDataset(object):
    """docstring for plot_correlation_windows"""

    def __init__(self, inputfile):
        super(CCDataset, self).__init__()
        self.station_pair = os.path.splitext(os.path.basename(inputfile))[0]
        self.datafile = h5py.File(inputfile, 'r')
        self.filekeys = list(self.datafile['corr_windows'].keys())
        self.data = None
        self.datakeys = []

        self.fs = dict(self.datafile['stats'].attrs)['sampling_rate']
        self.npts = self.datafile['corr_windows'][self.filekeys[0]].shape[0]
        self.max_lag = (self.npts - 1) / 2 / self.fs
        self.lag = np.linspace(-self.max_lag, self.max_lag, self.npts)
        self.ntraces = len(self.filekeys)

        self.ref = np.zeros(self.npts)
        self.stacks = {}
        self.refs = {}

    def __str__(self):
        output = "" 
        output += "Cross-correlation dataset:\n"
        output += "Maximum lag: {} seconds\n".format(self.max_lag)
        output += "Sampling rate: {} Hz\n".format(self.fs)
        output += "Number of traces: {}\n".format(len(self.filekeys))
        return(output)

    def data_to_memory(self, n_corr_max=None):
        
        if n_corr_max is None:
            n_corr_max = len(self.filekeys)
        try:
            self.data = np.zeros((n_corr_max, self.npts))
        except MemoryError:
            print("Data doesn't fit in memory, try setting a lower n_corr_max")
            return()

        for i, (k, v) in enumerate(self.datafile["corr_windows"].items()):
            if i == n_corr_max:
                break
            self.data[i, :] = v[:]
            self.datakeys.append(k)

    def data_to_dataframe(self, lag0, lag1, n_corr_max=None, normalize=False):
        ixs_get0 = np.where(self.lag >= lag0)
        ixs_lets = np.where(self.lag <= lag1)
        ixs = np.intersect1d(ixs_get0, ixs_lets)
        self.df_lags = self.lag[ixs]
        nr_samples = len(ixs)
        if n_corr_max is None:
            n_corr_max = self.ntraces
        self.df = pd.DataFrame(columns=range(nr_samples), 
                               data=np.zeros((n_corr_max, nr_samples)))
        for i in range(n_corr_max):
            for j in range(nr_samples):
                if normalize:
                    self.df.iat[i, j] = self.data[i, ixs][j] / (self.data[i, ixs].max() + np.finfo(float).eps)
                else:
                    self.df.iat[i, j] = self.data[i, ixs] / self.data[i, ixs][j]


            
    def data_from_mem(self, ix=None, k=None):
        if ix is None:
            return self.data[datakeys.index(k), :]
        else:
            return self.data[ix, :]

    def data_from_file(self, ix):
        return(self.datafile["corr_windows"][self.filekeys[ix]][:])

    def stack_by_cluster(self, stack_length="daily", from_mem=True, cluster=0):

        self.stack_length = stack_length

        if from_mem:
            get_data = self.data_from_mem
        else:
            get_data = self.data_from_file

        previous = ""
        previous_hour = 00
        stacks = []
        stack_start_times = []

        ix_cluster = np.where(self.labels == cluster)[0]
        print(ix_cluster)
        for ix in ix_cluster:#, k in enumerate(self.filekeys[ix_cluster]):
            k = self.filekeys[ix]
            if from_mem and ix == self.data.shape[0]:

                if "stack" in locals():
                    stack = Trace(np.array(stack) / stack_count)
                    stack.stats.sampling_rate = self.fs
                    stack.stats.sac = {}
                    stack.stats.sac["b"] = '{},{},{},{},{}'.format(*k.split('.')[0: 5])
                    stacks.append(stack)
                break
            if stack_length == 'daily':
                timestr = '{}.{}'.format(*k.split('.')[0: 2])
                if timestr != previous:
                    if previous != "":
                        stack = Trace(np.array(stack) / stack_count)
                        stack.stats.sampling_rate = self.fs
                        stack.stats.sac = {}
                        stack.stats.sac["b"] = '{},{},{},{},{}'.format(*k.split('.')[0: 5])
                        stacks.append(stack)

                    previous = timestr
                    stack = get_data(ix)
                    stack_count = 1
                else:
                    stack += get_data(ix)
                    stack_count += 1

        self.stacks = stacks
        print("Compiled {} stacks".format(len(self.stacks)))

    def linear_stack(self, stack_length="daily", from_mem=True):

        self.stack_length = stack_length

        if from_mem:
            get_data = self.data_from_mem
        else:
            get_data = self.data_from_file

        previous = ""
        previous_hour = 00
        stacks = []
        stack_start_times = []
        for ix, k in enumerate(self.filekeys):
            if from_mem and ix == self.data.shape[0]:
                if "stack" in locals():
                    stack = Trace(np.array(stack) / stack_count)
                    stack.stats.sampling_rate = self.fs
                    stack.stats.sac = {}
                    stack.stats.sac["b"] = '{},{},{},{},{}'.format(*k.split('.')[0: 5])
                    stacks.append(stack)
                break
            if stack_length == 'daily':
                timestr = '{}.{}'.format(*k.split('.')[0: 2])
                if timestr != previous:
                    if previous != "":
                        stack = Trace(np.array(stack) / stack_count)
                        stack.stats.sampling_rate = self.fs
                        stack.stats.sac = {}
                        stack.stats.sac["b"] = '{},{},{},{},{}'.format(*k.split('.')[0: 5])
                        stacks.append(stack)

                    previous = timestr
                    stack = get_data(ix)
                    stack_count = 1
                else:
                    stack += get_data(ix)
                    stack_count += 1

            elif stack_length[-6: ] == "hourly":
                try:
                    hourstep = hour_strings[stack_length[: -6]]
                except KeyError:
                    raise ValueError("x-hourly stack duration can be: ",
                                    (len(hour_strings) * "{}, ").format(*[k in hour_strings.keys()]))
                daystr = "{}.{}".format(*k.split(".")[0: 2])
                hourstr = k.split(".")[2]
                
                if float(hourstr) - float(previous_hour) > hourstep or daystr != previous:
                    if "stack" in locals():
                        stack = Trace(np.array(stack) / stack_count)
                        stack.stats.sampling_rate = self.fs
                        stack.stats.sac = {}
                        stack.stats.sac["b"] = '{},{},{},{},{}'.format(*k.split('.')[0: 5])
                        stacks.append(stack)

                    stack = get_data(ix)
                    stack_count = 1
                    previous_hour = hourstr
                    previous = daystr
                else:
                    if "stack" in locals():
                        stack += get_data(ix)
                        stack_count += 1
                    else:
                        stack = get_data(ix)
                        stack_count = 1

        self.stacks = stacks
        print("Compiled {} stacks".format(len(self.stacks)))


    def reference(self, reftype):

        if reftype == "arithmetic_mean":

            if self.data is not None:
                self.ref = np.mean(self.data, axis=0)

        elif reftype == "median":
            if self.data is not None:
                self.ref = np.median(self.data, axis=0)



    def filter_stacks(self, taper_perc=0.05, filter_type="bandpass",
                      f_hp=None, f_lp=None, corners=4, zerophase=True):

        for k, tr in self.stacks.items():
            for t in tr:
                t.taper(taper_perc)
                t.filter(filter_type, freqmin=f_hp,
                         freqmax=f_lp, corners=corners, zerophase=zerophase)


    def filter_data(self, taper_perc=0.05, filter_type="bandpass",
                    f_hp=None, f_lp=None, corners=4, zerophase=True):

        if filter_type == 'bandpass':
            sos = filter.bandpass(df=self.fs, freqmin=f_hp, freqmax=f_lp,
                                  corners=corners)
        elif filter_type == 'lowpass':
            sos = filter.lowpass(df=self.stats['Fs'], freq=f_lp, corners=corners)
        elif filter_type == 'highpass':
            sos = filter.highpass(df=self.stats['Fs'], freq=f_hp, corners=corners)
        else:
            msg = 'Filter %s is not implemented, implemented filters:\
            bandpass, highpass,lowpass' % type
            raise ValueError(msg)

        for i, tr in enumerate(self.data):

            if zerophase:
                firstpass = sosfilt(sos, tr)
                self.data[i, :] = sosfilt(sos, firstpass[::-1])[::-1]
                # then assign to newfile, which might be self.file
            else:
                self.data[i, :] = sosfilt(sos, tr)


    def get_window(self, t_mid, hw, window_type="hann"):

        half_lag_minus_one = int((self.npts - 1) / 2)
        ix_1 = np.argmin(abs(self.lag[self.lag > 0] - t_mid - hw))
        ix_0 = np.argmin(abs(self.lag[self.lag > 0] - t_mid + hw))
        win = np.zeros(self.lag.shape)
        win[half_lag_minus_one + ix_0 : half_lag_minus_one + ix_1] += hann(ix_1 - ix_0)
        return(win)


    def window_stacks(self, t_mid, hw):
        
        win = self.get_window(t_mid, hw)

        for stack in self.stacks:
            stack.data *= win


    def window_data(self, t_mid, hw):
        win = self.get_window(t_mid, hw)
        for ix in range(self.data.shape[0]):
            self.data[ix, :] *= window

    def interpolate_stacks(self, new_fs):

        new_npts = int(self.npts / self.fs * new_fs)
        if new_npts % 2 == 0:
            new_npts += 1

        new_lag = np.linspace(-self.max_lag, self.max_lag, new_npts)
        for stack in self.stacks:
            f = interp1d(self.lag, stack.data, kind="cubic")
            stack.data = f(new_lag)
        self.lag = new_lag
        self.npts = new_npts
        self.fs = (2 * self.max_lag + 1. / new_fs) / self.npts

    def form_clusters(self, n_clusters, lag0=0, lag1=20, n_corr_max=None, normalize=False):
        self.data_to_dataframe(lag0=lag0, lag1=lag1, n_corr_max=n_corr_max,
                                normalize=normalize)
        self.labels, self.centroids = cluster(self.df, n_clusters)

    def plot_clustered_data(self, by_cluster=True, by_date=False, plotting_scale=0.1):
        if "labels" not in self.__dict__.keys():
            raise ValueError("Must cluster the data using cluster_data before plotting")
        import seaborn as sns

        n_clusters = len(self.centroids)
        nr_samples = len(self.df.columns)
        current_palette = sns.color_palette("bright", n_clusters + 1)
        colors = list(map(lambda x: current_palettform_cle[x+1], self.labels))
        if by_date:
            fig = plt.figure(figsize=(5, 15))
            for i in range(len(self.df)):
                tsplot = np.zeros(nr_samples)
                for j in range(nr_samples):
                    tsplot[j] = self.df.iat[i, j]
                plt.plot(self.df_lags, tsplot / tsplot.max() + i, color=colors[i], alpha=0.5)
            
        if by_cluster:
            fig = plt.figure(figsize=(5, 15))
            offset = 0
            for l in set(self.labels):
                color = current_palette[l]
                ixn = np.where(self.labels == l)
                tsplot = np.zeros(nr_samples)
                for i in ixn[0]:
                    for j in range(nr_samples):
                        tsplot[j] = self.df.iat[i, j]
                    plt.plot(self.df_lags, tsplot / tsplot.max() + offset * plotting_scale, color=color, alpha=0.5)
                    offset += 1
                offset += 2
        plt.xlabel("Lag (s)")
        plt.yticks([])
        plt.title("K-means clustering of traces from:\n{}\n{} clusters".format(self.station_pair,
                                                                               len(self.centroids)))
        plt.savefig("clusters.png")
        plt.show()

    def plot_stacks(self, outfile=None, seconds_to_show=20, scale_factor_plotting=0.1,
        plot_mode="traces", seconds_to_start=0.0, cmap=plt.cm.bone):
        if len(self.stacks) == 0:
            return()
        ylabels = []
        ylabelticks = []
        if plot_mode == "traces":
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cnt = 0
            for t in self.stacks:
                ax.plot(self.lag, t.data / t.data.max() + scale_factor_plotting * cnt,
                        'k', alpha=0.5, linewidth=0.5)
                cnt += 1
                if UTCDateTime(t.stats.sac["b"]).strftime("%d") == "01"\
                   and UTCDateTime(t.stats.sac["b"]).strftime("%d.%m.%y") not in ylabelticks:
                    ylabels.append(scale_factor_plotting * cnt)
                    ylabelticks.append(UTCDateTime(t.stats.sac["b"]).strftime("%d.%m.%y"))
            
            if self.ref.sum() != 0:
                ax.plot(self.lag, self.ref / self.ref.max(), linewidth = 2.0)
                plt.ylim([-1, scale_factor_plotting * cnt + np.mean(abs(t.data))])
            else:
                plt.ylim([0 - np.mean(abs(t.data)), 
                          scale_factor_plotting * cnt + np.mean(abs(t.data))])
            plt.title(self.station_pair)

        elif plot_mode == "heatmap":
            fig = plt.figure()
            ax = fig.add_subplot(111)
            dat_mat = np.zeros((len(self.stacks), self.npts))
            for ix, t in enumerate(self.stacks):
                dat_mat[ix, :] = t.data
                if UTCDateTime(t.stats.sac["b"]).strftime("%d") == "01"\
                    and UTCDateTime(t.stats.sac["b"]).strftime("%d.%m.%y") not in ylabelticks:
                    ylabels.append(ix)
                    ylabelticks.append(UTCDateTime(t.stats.sac["b"]).strftime("%d.%m.%y"))
            plt.pcolormesh(self.lag, range(len(self.stacks)), dat_mat,
                           vmax=scale_factor_plotting*dat_mat.max(),
                           vmin=-scale_factor_plotting*dat_mat.max(),
                           cmap=cmap)
                
                
        plt.ylabel("Normalized {} stack".format(self.stack_length))
        plt.xlim([seconds_to_start, seconds_to_show])
        plt.xlabel("Lag (seconds)")
        plt.yticks(ylabels, ylabelticks)
        ax.yaxis.tick_right()

        plt.grid(linestyle=":", color="lawngreen", axis="x")
        if seconds_to_show - seconds_to_start > 20:
            tickstep = 2.0
        elif seconds_to_show - seconds_to_start > 10:
            tickstep = 1.0
        else:
            tickstep = 0.5
        plt.xticks([i for i in np.arange(seconds_to_start, seconds_to_show, tickstep)],
                   [str(i) for i in np.arange(seconds_to_start, seconds_to_show, tickstep)])
        if outfile is not None:
            plt.tight_layout()
            plt.savefig(outfile)
        
        plt.show()


# import sys
# fl = sys.argv[1]
# sec_to_show = float(sys.argv[2])
# scale_factor = float(sys.argv[3])
# plot_mode = sys.argv[4]
# try:
#     seconds_to_start = float(sys.argv[5])
# except IndexError:
#     seconds_to_start = 0.0
# a = plot_correlation_windows([fl])
# a.get_stacks()
# a.filter_stacks()
# a.plot_stacks(outfile='autocorr_daily_{}.png'.format(fl.split('.')[1]),
#         seconds_to_show=sec_to_show, scale_factor_plotting=scale_factor,
#         plot_mode=plot_mode, seconds_to_start=seconds_to_start)
