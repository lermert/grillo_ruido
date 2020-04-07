import numpy as np
import h5py
from obspy import Trace, UTCDateTime
import matplotlib.pyplot as plt
from scipy.signal import sosfilt
#from ruido.utils import filter
from utils import filter
import os

hour_strings = {"three": 3.0, "four": 4.0, "six": 6.0, "eight": 8.0, "twelve": 12.}

class CCDataset(object):
    """docstring for plot_correlation_windows"""

    def __init__(self, inputfile):
        super(CCDataset, self).__init__()
        self.station_pair = os.path.splitext(os.path.basename(inputfile))[0]
        self.datafile = h5py.File(inputfile, 'r')
        self.datakeys = list(self.datafile['corr_windows'].keys())
        self.data = None

        self.fs = dict(self.datafile['stats'].attrs)['sampling_rate']
        self.npts = self.datafile['corr_windows'][self.datakeys[0]].shape[0]
        self.max_lag = (self.npts - 1) / 2 / self.fs
        self.lag = np.linspace(-self.max_lag, self.max_lag, self.npts)

        self.ref = np.zeros(self.npts)
        self.stacks = {}
        self.refs = {}

    def __str__(self):
        output = "" 
        output += "Cross-correlation dataset:\n"
        output += "Maximum lag: {} seconds\n".format(self.max_lag)
        output += "Sampling rate: {} Hz\n".format(self.fs)
        output += "Number of traces: {}\n".format(len(self.datakeys))
        return(output)

    def data_to_memory(self, n_corr_max=None):
        
        if n_corr_max is None:
            n_corr_max = len(self.datakeys)
        try:
            self.data = np.zeros((n_corr_max, self.npts))
        except MemoryError:
            print("Data doesn't fit in memory, try setting a lower n_corr_max")
            return()

        for i, (k, v) in enumerate(self.datafile["corr_windows"].items()):
            if i == n_corr_max:
                break
            self.data[i, :] = v[:]


    def data_from_mem(self, ix):
        return self.data[ix, :]

    def data_from_file(self, ix):
        return(self.datafile["corr_windows"][self.datakeys[ix]][:])

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
        for ix, k in enumerate(self.datakeys):
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
            # for k, tr in self.stacks.items():
            #     all_stack = np.zeros(tr[0].data.shape)
            #     for t in tr:
            #         all_stack += t.data
            #         cnt += 1
            #     self.refs[k] = all_stack
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



    def plot_stacks(self, outfile=None, seconds_to_show=20, scale_factor_plotting=0.1,
        plot_mode="traces", seconds_to_start=0.0, cmap=plt.cm.bone):
        
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
                if UTCDateTime(t.stats.sac["b"]).strftime("%d") == "01" or cnt == 0:
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
                if UTCDateTime(t.stats.sac["b"]).strftime("%d") == "01" or ix == 0:
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
        plt.xticks([i for i in np.arange(seconds_to_start, seconds_to_show, 0.5)],
                   [str(i) for i in np.arange(seconds_to_start, seconds_to_show, 0.5)])
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
