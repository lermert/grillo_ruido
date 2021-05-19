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


dirs = ["/media/lermert/Bienenstich/data_CDMX/correlations/UNM_seis_ccc/*_raw/"]
channels = ["BHZ", "BHN", "BHE", "HHE", "HHN", "HHZ"]
corrT = 1200.
duration = 10 * 86400.
step = 10 * 86400.
minimum_stack_len = 1
t0 = UTCDateTime("1995,01,01").timestamp
t1 = UTCDateTime("2021,02,01").timestamp
event_dates = [UTCDateTime("2008,184").timestamp, UTCDateTime("2013,046").timestamp, UTCDateTime("2017,251").timestamp,
               UTCDateTime("2017,262").timestamp, UTCDateTime("2020,090").timestamp]
cmap = cm.bilbao
stackmode = "linear"   # linear, median, robust
fmin = 0.05
fmax = 10.0
vmaxs = {"BH": -95, "HH": -75, "HL": -75, "HN":-90}
vmins = {"BH": -130, "HH": -125, "HL": -125, "HN":-125}
do_stacking = True


def add_stacks(dset):
    if len(dset.dataset) == 1:
        t_running = max(t0, dset.dataset[0].timestamps.min())
    else:
        t_running = max(t0, dset.dataset[0].timestamps.min(), dset.dataset[1].timestamps.max())

    while t_running < min(t1, dset.dataset[0].timestamps.max()):
        print(UTCDateTime(t_running).strftime("%Y/%m/%d"), end=",")
        stimes = dset.group_for_stacking(t_running, duration=duration)

        if stimes == []:
            # print(t_running, "no windows:)
            t_running += step
            continue
        if len(stimes) < minimum_stack_len:
            t_running += step
            continue

        dset.stack(np.array(stimes), stackmode=stackmode, epsilon_robuststack=None)
        t_running += step


for indir in dirs:
    for cha in channels:
        input_pattern = dir + "/*{}*{}*".format(cha, cha)
        input_files = glob(input_pattern)
        if input_files == []:
            continue

        input_files.sort()
        vmin = vmins[cha[0:2]]
        vmax = vmaxs[cha[0:2]]

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
            cwin.create_dataset("timestamps", shape=dset.dataset[1].timestamps.shape, dtype=dtp)
            for ixttsp, tstmp in enumerate(dset.dataset[1].timestamps):
                cwin["timestamps"][ixttsp] = UTCDateTime(tstmp).strftime("%Y.%j.%H.%M.%S")
            outfile.flush()
            outfile.close()


        print(dset.dataset[1].fs)
        station = os.path.basename(input_files[0]).split()[1]
        taper = cosine_taper(npts=dset.dataset[1].npts, p=0.1)
        # once all the stacks, take the spectra
        #if rank == 0:
        freq = np.fft.rfftfreq(dset.dataset[1].data.shape[-1] * 4, d=1./dset.dataset[1].fs)
        ix0 = np.argmin((freq - fmin)**2)
        ix1 = np.argmin((freq - fmax * 1.1)**2)
        specgram = np.zeros((dset.dataset[1].data.shape[0], freq.shape[0]))
        for i in range(dset.dataset[1].data.shape[0]):

            specgram[i, :] = np.abs(np.fft.rfft(taper * dset.dataset[1].data[i, :], n=dset.dataset[1].data.shape[-1] * 4))
            # uncomment the line below if data was in velocity
            # specgram[i, :] *= (2. * np.pi * freq) ** 2
            freq[0] = freq[1]
            if i % 3 == 0:
                plt.loglog(freq, specgram[i, :], color="0.5", alpha=0.1, linewidth=0.5)  # color=str(i/len(dset.dataset[1].timestamps)),
        plt.ylim(vmin, vmax)
        plt.grid()
        plt.savefig("spectraldensity.png")


        fig = plt.figure(figsize=(8, 4.5))
        x, y = np.meshgrid(np.arange(dset.dataset[1].timestamps.shape[0]), freq[ix0: ix1])
        plt.pcolor(x, y, 10 * np.log10(specgram[:, ix0: ix1].T), cmap=cmap,
                  vmin=vmin, vmax=vmax)
        plt.colorbar(label="dB (to 1 m\u00B2 s\u207B\u2074 Hz\u207B\u00B9)")

        for ev in event_dates:
            ixtev = np.argmin((dset.dataset[1].timestamps - 0.5*duration - ev) ** 2)
            plt.scatter([ixtev], [9], color="lightblue", marker="v")

        years = []
        xticks = []
        for ixt, tt in enumerate(dset.dataset[1].timestamps):
            if UTCDateTime(tt).strftime("%Y") not in years:
                years.append(UTCDateTime(tt).strftime("%Y"))
                xticks.append(ixt)
        #plt.yticks(np.arange(10), [str(hz) for hz in range(10)])
        plt.ylim(fmin, fmax)
        plt.xticks(xticks, years, rotation=30)
        plt.ylabel("Frequency (Hz)")
        plt.tight_layout()
        plt.savefig("spectrogram_{}_{}_{}-{}Hz.png".format(station, cha, fmin, fmax), dpi=300)
