import numpy as np
from ruido.cc_dataset_newnewnew import CCDataset, CCData
from obspy import UTCDateTime
import os
import numpy as np
import time
from glob import glob
from cmcrameri import cm 
import matplotlib.pyplot as plt
import h5py
import re
from obspy.signal.invsim import cosine_taper
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


dirs = ["TO_ccc/TO_*_ccc/", "RSVM/rsvm_*/", "UNM_seis_ccc/*_raw/"]
channels = ["HLE", "HLN", "HLZ", "BHZ", "BHN", "BHE", "HHE", "HHN", "HHZ"]
stations = ["UNM", "AOVM", "APVM", "BJVM",
            "MCVM", "MHVM", "PBVM",
            "CJVM", "COVM", "CTVM", "GMVM",
            "ICVM", "MPVM", "MZVM", "PTVM",
            "VRVM", "THVM", "TLVM", "XCVM",
            "CIRE", "ESTA", "MULU", "TEPE", "MIXC"]
corrT = 1200.
duration = 5 * 86400.
step = 5 * 86400.
minimum_stack_len = 1
t0 = UTCDateTime("1995,01,01").timestamp
t1 = UTCDateTime("2021,02,01").timestamp
event_dates = [UTCDateTime("2008,184").timestamp, UTCDateTime("2013,046").timestamp, UTCDateTime("2017,251").timestamp,
               UTCDateTime("2017,262").timestamp, UTCDateTime("2020,090").timestamp]
plot_type = "spectrogram"  # spectrogram or periodogram
cmap = cm.bilbao
stackmode = "linear"   # linear, median, robust
fmin = 0.05
fmax = 10.0
Tmin = 0.1
Tmax = 12.0
vmaxs = {"BH": -95, "HH": -75, "HL": -75, "HN":-90}
vmins = {"BH": -130, "HH": -125, "HL": -125, "HN":-125}
datesplot0 = {"BH": "1995,001", "HH": "2005,170", "HL": "2017,100" }
datesplot1 = {"BH": "2021,001", "HH": "2007,170", "HL": "2021,001" }
do_stacking = True


def add_stacks(dset):
    if len(dset.dataset) == 1:
        t_running = max(t0, dset.dataset[0].timestamps.min())
    else:
        t_running = max(t0, dset.dataset[0].timestamps.min(), dset.dataset[1].timestamps.max())

    while t_running < min(t1, dset.dataset[0].timestamps.max()):
        print(UTCDateTime(t_running).strftime("%Y/%m/%d"), end=",")
        stimes = dset.group_for_stacking(t_running, duration=duration)

        if stimes == [] or len(stimes) < minimum_stack_len:
            dset.dataset[1].extend(new_data=np.zeros((1, dset.dataset[1].npts)), new_timestamps=[t_running],
                                   new_fs=dset.dataset[1].fs, keep_duration=-1)
            t_running += step
            continue

        dset.stack(np.array(stimes), stackmode=stackmode, epsilon_robuststack=None)
        t_running += step


for indir in dirs:
    for cha in channels:
        for sta in stations:
            input_pattern = indir + "/*{}*{}*{}*.h5".format(sta, cha, cha)
            input_files = glob(input_pattern)
            if input_files == []:
                continue

            input_files.sort()
            vmin = vmins[cha[0:2]]
            vmax = vmaxs[cha[0:2]]
            dateplot0 = UTCDateTime(datesplot0[cha[0:2]]).timestamp
            dateplot1 = UTCDateTime(datesplot1[cha[0:2]]).timestamp

            # read in the data, one file at a time, adding stacks as we go along
            if do_stacking:
                dset = CCDataset(input_files[0])
                for i, f in enumerate(input_files):
                    if i == 0:
                        dset.data_to_memory(n_corr_max=None)
                    else:
                        dset.add_datafile(f)
                        dset.data_to_memory(keep_duration=duration * 1.5)

                    #if rank == 0:
                    print("Read")
                    #dset.dataset[0].lose_allzero_windows()
                    dset.dataset[0].add_rms()
                    add_stacks(dset)
                    # else:
                        # pass
            else:
                dset = CCDataset(input_files[0])
                dset.data_to_memory()
                dset.dataset[1] = CCData(dset.dataset[0].data.copy(), dset.dataset[0].timestamps.copy(), dset.dataset[0].fs)


            if do_stacking:
                dset.dataset[1].data *= (2. / dset.dataset[1].fs /(corrT * dset.dataset[1].fs))

            if do_stacking:
                outfile = re.sub("windows", "allwin_stacks", os.path.basename(input_files[0]))
                outfile = h5py.File(outfile, "w")
                cwin = outfile.create_group("corr_windows")
                stats = outfile.create_dataset("stats", data=())
                stats.attrs["channel1"] = cha
                stats.attrs["channel2"] = cha
                stats.attrs["distance"] = 0.0
                stats.attrs["sampling_rate"] = dset.dataset[1].fs
                stats.attrs["duration"] = duration
                stats.attrs["step"] = step
                stats.attrs["minimum_stack_len"] = minimum_stack_len
                stats.attrs["freq_band"] = "all"
                stats.attrs["filter_type"] = "none"
                stats.attrs["t0"] = t0
                stats.attrs["t1"] = t1
                stats.attrs["rms_percentile"] = "none"
                stats.attrs["stackmode"] = stackmode
                cwin.create_dataset("data", data=dset.dataset[1].data)

                dtp = h5py.string_dtype()
                cwin.create_dataset("timestamps",
                                    shape=dset.dataset[1].timestamps.shape,
                                    dtype=dtp)
                for ixttsp, tstmp in enumerate(dset.dataset[1].timestamps):
                    cwin["timestamps"][ixttsp] = \
                        UTCDateTime(tstmp).strftime("%Y.%j.%H.%M.%S")
                outfile.flush()
                outfile.close()

            print(dset.dataset[1].fs)
            taper = cosine_taper(npts=dset.dataset[1].npts, p=0.1)
            freq = np.fft.rfftfreq(dset.dataset[1].data.shape[-1] * 4,
                                   d=1./dset.dataset[1].fs)
            freq[0] = freq[1]

            if plot_type == "spectrogram":
                ix0 = np.argmin((freq - fmin)**2)
                ix1 = np.argmin((freq - fmax * 1.1)**2)
            elif plot_type == "periodogram":
                period = 1. / freq
                ix0p = np.argmin((freq - 1. / Tmax) ** 2)
                ix1p = np.argmin((freq - 1. / Tmin) ** 2)
            else:
                raise ValueError("Unknown plot_type {}".format(plot_type))

            specgram = np.zeros((dset.dataset[1].data.shape[0], freq.shape[0]))
            if plot_type == "periodogram":
                specgramplot = np.zeros((dset.dataset[1].data.shape[0],
                                         ix1p - ix0p))

            for i in range(dset.dataset[1].data.shape[0]):
                specgram[i, :] = np.abs(np.fft.rfft(taper *
                                        dset.dataset[1].data[i, :],
                                        n=dset.dataset[1].data.shape[-1] * 4))
                if plot_type == "periodogram":
                    specgramplot[i, :] = specgram[i, ix0p: ix1p].copy()
                    specgramplot[i, :] = specgramplot[i, ::-1]

            fig = plt.figure(figsize=(8, 4.5))

            if plot_type == "periodogram":
                xplot = period[ix0p: ix1p].copy()
                xplot = xplot[::-1]
            else:
                xplot = freq[ix0: ix1]
            x, y = np.meshgrid(dset.dataset[1].timestamps, xplot)

            if plot_type == "periodogram":
                plt.pcolor(x, y, 10 * np.log10(specgramplot).T, cmap=cmap,
                           vmin=vmin, vmax=vmax)
            elif plot_type == "spectrogram":
                plt.pcolor(x, y, 10 * np.log10(specgram[:, ix0: ix1].T),
                           cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(label="dB (to 1 m\u00B2 s\u207B\u2074 Hz\u207B\u00B9)")

            for ev in event_dates:
                ixtev = np.argmin((dset.dataset[1].timestamps - 0.5 * duration - ev) ** 2)
                if plot_type == "periodogram":
                    plt.scatter(dset.dataset[1].timestamps[ixtev], [0.9 * Tmax],
                                color="k", marker="v")
                else:
                    plt.scatter(dset.dataset[1].timestamps[ixtev], [0.9 * fmax],
                                color="k", marker="v")

            years = []
            yearsyears = []
            xticks = []
            for ixt, tt in enumerate(dset.dataset[1].timestamps):
                if UTCDateTime(tt).strftime("%Y") not in yearsyears:
                    if tt < dateplot0:
                        years.append(UTCDateTime(dateplot0).strftime("%Y/%m"))
                    elif tt > dateplot1:
                        years.append(UTCDateTime(dateplot1).strftime("%Y/%m"))
                    else:
                        years.append(UTCDateTime(tt).strftime("%Y/%m"))
                    yearsyears.append(UTCDateTime(tt).strftime("%Y"))
                    xticks.append(tt)

            if plot_type == "periodogram":
                plt.ylim(Tmin, Tmax*0.9)
                plt.ylabel("Period (s)")
                outfile = "periodogram_{}_{}_{}-{}s.png".format(sta, cha, Tmin, Tmax)

            else:
                plt.ylim(fmin, fmax*0.9)
                plt.ylabel("Frequency (Hz)")
                outfile = "spectrogram_{}_{}_{}-{}Hz.png".format(sta, cha, fmin, fmax)
            plt.xlim(dateplot0, dateplot1)
            plt.xticks(xticks, years, rotation=30)
            plt.tight_layout()
            plt.savefig(outfile, dpi=300)
