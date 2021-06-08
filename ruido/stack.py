# coding: utf-8
from ruido.cc_dataset_mpi import CCDataset, CCData
from obspy import UTCDateTime
from obspy.signal.filter import envelope
from obspy.signal.invsim import cosine_taper
from scipy.signal import find_peaks
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import h5py
from glob import glob
from warnings import warn
from mpi4py import MPI
from cmcrameri import cm
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
# =================================
# INPUT
# =================================
input_file_basename = "/media/lermert/Ablage/corrs_from_workstation/G.UNM.*.{}--G.UNM.*.{}.ccc.windows.h5"
output_file_basename = "G.UNM.{}--G.UNM.{}.ccc.{}.h5"
years = range(2006, 2022)
# years = ["../datasets/"]
comp_pairs =[["BHZ", "BHN"], ["BHZ", "BHZ"], ["BHN", "BHE"], ["BHN", "BHN"], ["BHE", "BHE"]]
station = "UNM"  # for metadata table
freq_bands = [[0.25, 0.5], [0.5, 1.0], [1.0, 2.0], [2., 4.], [4.0, 8.0]]
twins_plot = [[-60, 60], [-40., 40.], [-20., 20.], [-10., 10.], [-6., 6.]]
scale_factor_plotting = 1.0
colormap = cm.bilbao
plotlabel = "year"
duration = 10. * 86400.
step = 10. * 86400.
minimum_stack_len = 100
t0 = UTCDateTime("1995,01,01").timestamp
t1 = UTCDateTime("2021,02,01").timestamp
filter_type = "cheby2_bandpass"
filter_maxord = 12
percentile_rms = 75
stackmode = "linear"   # linear, median, robust
robuststack_epsilon = 1.e-3
use_clusters = True
clusterdir = "results_from_uwork"
# =================================

# Script:
def add_stacks(dset, t_running_in=None):

    # make a difference whether there are cluster labels or not.
    # if there are then use them for selection.

    if dset.dataset[0].cluster_labels is not None:

        for clabel in np.unique(dset.dataset[0].cluster_labels):
            print("cluster ", clabel)
            if clabel == -1:   # the unmatched timestamps
                warn("Unmatched timestamps present, is this ok?")
                continue

            if t_running_in is None:
                t_running = max(t0, dset.dataset[0].timestamps.min())
            else:
                t_running = t_running_in

            while t_running < min(t1, dset.dataset[0].timestamps.max()):

                stimes = dset.group_for_stacking(t_running, duration=duration,
                                                 cluster_label=clabel)
                stimes = dset.select_for_stacking(stimes, "rms_percentile",
                                                  bootstrap=False,
                                                  perc=percentile_rms,
                                                  mode="upper")
                if stimes == []:
                    print("No windows, ", UTCDateTime(t_running))
                    t_running += step
                    continue
                if len(stimes) < minimum_stack_len:
                    print("Not enough windows, ", UTCDateTime(t_running))
                    t_running += step
                    continue

                dset.stack(np.array(stimes), stackmode=stackmode,
                           epsilon_robuststack=robuststack_epsilon,
                           stacklevel_out=clabel+1)
                t_running += step

    else:
        if t_running is None:
            t_running = max(t0, dset.dataset[0].timestamps.min())

        while t_running < min(t1, dset.dataset[0].timestamps.max()):
            stimes = dset.group_for_stacking(t_running, duration=duration)
            stimes = dset.select_for_stacking(stimes, "rms_percentile", bootstrap=False,
                                              perc=percentile_rms, mode="upper")
            if stimes == []:
                # print(t_running, "no windows:)
                t_running += step
                continue
            if len(stimes) < minimum_stack_len:
                t_running += step
                continue

            dset.stack(np.array(stimes), stackmode=stackmode, epsilon_robuststack=robuststack_epsilon)
            t_running += step


# loop over components:
for cpair in comp_pairs:
    input_files = []
    #for yr in years:
    #    input_files.extend(glob(input_file_basename.format(yr, *cpair)))
    input_files.extend(glob(input_file_basename.format(*cpair)))
    
    # loop over frequency bands
    for ixf, freq_band in enumerate(freq_bands):

        if use_clusters:
            clusterfile = os.path.join(clusterdir,
                                       "{}.{}.{}.{}-{}Hz.gmmlabels.npy".format(
                                        station, cpair[0], cpair[1], freq_band[0],
                                        freq_band[1]))
            clusters = np.load(clusterfile)

        # track how long it takes
        trun = time.time()
        trun0 = trun

        # read in the data, one file at a time, adding stacks as we go along
        dset = CCDataset(input_files[0])

        for i, f in enumerate(input_files):
            if i == 0:
                dset.data_to_memory(n_corr_max=None)
                dset.dataset[0].add_cluster_labels(clusters)
            else:
                dset.add_datafile(f)
                dset.data_to_memory(keep_duration=3*duration)
                dset.dataset[0].add_cluster_labels(clusters)

            print(UTCDateTime(clusters[0, 0]), UTCDateTime(clusters[0, -1]))

            if rank == 0:
                print("Read ")
                print(time.time() - trun)
            else:
                pass

            dset.filter_data(f_hp=freq_band[0], f_lp=freq_band[1], taper_perc=0.2,
                             stacklevel=0, filter_type=filter_type,
                             maxorder=filter_maxord, npool=8)
            if rank == 0:
                print("Filtered ")
                print(time.time() - trun)

                try:
                    t_running = dset.dataset[1].timestamps.max() + step
                except KeyError:
                    t_running = max(dset.dataset[0].timestamps.min(), t0)
                add_stacks(dset, t_running)
                print("Stacked ")
                print(dset)
                print(time.time() - trun)
            else:
                pass

            comm.barrier()

             # plot, if intended
        if twins_plot is not None and rank == 0:
            outplot = os.path.splitext(os.path.basename(input_files[0]))[0] + "{}-{}Hz.stacks.png".format(*freq_band)
            dset.plot_stacks(outfile=outplot, seconds_to_start=twins_plot[ixf][0], seconds_to_show=twins_plot[ixf][1],
                             cmap=colormap, mask_gaps=True, step=step, scale_factor_plotting=scale_factor_plotting,
                             plot_envelope=False, normalize_all=True, label_style=plotlabel, stacklevel=list(dset.dataset.keys())[-1])

        # save the stacks
        if rank == 0:
            print(dset)

            for stacklevel in dset.dataset.keys():
                if stacklevel == 0:
                    continue
                outfile = output_file_basename.format(*cpair, 
                    "stacks_{}days".format(duration//86400) + UTCDateTime(t0).strftime("%Y") +\
                        "-" + UTCDateTime(t1).strftime("%Y") + "_bp" + str(ixf) + "_cl" + str(stacklevel))
                outfile = h5py.File(outfile, "w")
                cwin = outfile.create_group("corr_windows")
                stats = outfile.create_dataset("stats", data=())
                stats.attrs["channel1"] = station + cpair[0]
                stats.attrs["channel2"] = station + cpair[1]
                stats.attrs["distance"] = 0.0
                stats.attrs["sampling_rate"] = dset.dataset[1].fs
                stats.attrs["duration"] = duration
                stats.attrs["step"] = step
                stats.attrs["minimum_stack_len"] = minimum_stack_len
                stats.attrs["freq_band"] = freq_band
                stats.attrs["filter_type"] = filter_type
                if filter_type == "cheby2_bandpass":
                    stats.attrs["filter_maxord"] = filter_maxord
                stats.attrs["t0"] = t0
                stats.attrs["t1"] = t1
                stats.attrs["rms_percentile"] = percentile_rms
                stats.attrs["stackmode"] = stackmode
                if stackmode == "robust":
                    stats.attrs["robuststack_epsilon"] = robuststack_epsilon
                cwin.create_dataset("data", data=dset.dataset[stacklevel].data)

                dtp = h5py.string_dtype()
                cwin.create_dataset("timestamps", shape=dset.dataset[stacklevel].timestamps.shape, dtype=dtp)
                for ixttsp, tstmp in enumerate(dset.dataset[stacklevel].timestamps):
                    cwin["timestamps"][ixttsp] = UTCDateTime(tstmp).strftime("%Y.%j.%H.%M.%S")
                outfile.flush()
                outfile.close()
