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
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
# =================================
# INPUT
# =================================
input_file_basename = "{}/G.UNM.*.{}--G.UNM.*.{}.pcc.windows.h5"
output_file_basename = "G.UNM.{}--G.UNM.{}.pcc.{}.h5"
#years = range(1995, 2022)
years = ["../datasets/"]
comp_pairs = [["HNZ", "HNZ"]]#[["BHZ", "BHE"], ["BHZ", "BHN"], ["BHZ", "BHZ"], ["BHN", "BHE"]]
station = "G.UNM"  # for metadata table
freq_bands = [[0.5, 1.0], [1.0, 2.0], [2., 4.]]
twins_plot = [[-40., 40.], [-20., 20.], [-10., 10.],]
scale_factor_plotting = 1.0
colormap = plt.cm.bone
plotlabel = "year"
duration = 3 * 86400.
step = 3 *  86400.
minimum_stack_len = 100
t0 = UTCDateTime("1995,01,01").timestamp
t1 = UTCDateTime("2021,02,01").timestamp
filter_type = "cheby2_bandpass"
filter_maxord = 18
percentile_rms = 50
stackmode = "linear"   # linear, median, robust
robuststack_epsilon = 1.e-3
# =================================

# Script:
def add_stacks(dset):

    print(dset)
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
    for yr in years:
        input_files.extend(glob(input_file_basename.format(yr, *cpair)))

    
    # loop over frequency bands
    for ixf, freq_band in enumerate(freq_bands):
        # track how long it takes
        trun = time.time()
        trun0 = trun

        # read in the data, one file at a time, adding stacks as we go along
        dset = CCDataset(input_files[0])
        for i, f in enumerate(input_files):
            if i == 0:
                dset.data_to_memory(n_corr_max=None)

                if rank == 0:
                    print("Read ")
                    print(time.time() - trun)
                    dset.dataset[0].lose_allzero_windows()
                    print("Removed zeros ")
                    print(time.time() - trun)
                else:
                    pass
                dset.filter_data(f_hp=freq_band[0], f_lp=freq_band[1], taper_perc=0.2,
                                 stacklevel=0, filter_type=filter_type,
                                 maxorder=filter_maxord, npool=8)
                if rank == 0:
                    print("Filter ")
                    print(time.time() - trun)
                    dset.dataset[0].add_rms()
                    add_stacks(dset)
                    print("stacked ")
                    print(time.time() - trun)
                else:
                    pass

                comm.barrier()

            else:
                dset.add_datafile(f)
                dset.data_to_memory(keep_duration=duration)

                if rank == 0:
                    dset.dataset[0].lose_allzero_windows()
                else:
                    pass
                dset.filter_data(f_hp=freq_band[0], f_lp=freq_band[1], taper_perc=0.2,
                                 stacklevel=0, filter_type=filter_type,
                                 maxorder=filter_maxord, npool=8)
                if rank == 0:
                    dset.dataset[0].add_rms()
                    add_stacks(dset)
                else:
                    pass
                comm.barrier()


             # plot, if intended
        if twins_plot is not None and rank == 0:
            outplot = os.path.splitext(os.path.basename(input_files[0]))[0] + "{}-{}Hz.stacks.png".format(*freq_band)
            dset.plot_stacks(outfile=outplot, seconds_to_start=twins_plot[ixf][0], seconds_to_show=twins_plot[ixf][1],
                             cmap=colormap, mask_gaps=True, step=step, scale_factor_plotting=scale_factor_plotting,
                             plot_envelope=False, normalize_all=True, label_style=plotlabel)

        # save the stacks
        if rank == 0:
            print(dset)
            outfile = output_file_basename.format(*cpair, "stacks_" + UTCDateTime(t0).strftime("%Y") +\
                    "-" + UTCDateTime(t1).strftime("%Y") + "_" + str(ixf))
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
            cwin.create_dataset("data", data=dset.dataset[1].data)

            dtp = h5py.string_dtype()
            cwin.create_dataset("timestamps", shape=dset.dataset[1].timestamps.shape, dtype=dtp)
            for ixttsp, tstmp in enumerate(dset.dataset[1].timestamps):
                cwin["timestamps"][ixttsp] = UTCDateTime(tstmp).strftime("%Y.%j.%H.%M.%S")
            outfile.flush()
            outfile.close()