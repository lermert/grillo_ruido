from ruido.cc_dataset_mpi import CCDataset, CCData
from obspy import UTCDateTime
import os
import numpy as np
import time
from glob import glob
import re
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

input_files = glob("UNM_BH_30d_raw_unaligned/G*.h5")
input_files.sort()

measurement_type = "stretching"

new_fs = 160.0
ngrid = 50
ngrid = int(ngrid)
plot_tmax = [100.0, 60.0, 40.0, 20.0, 20.0, 10.0, 10.0, 5.0]
skipfactor = 4  # allow this times more stretching as would be permitted to avoid cycle skipping
# note: this leads to cycle skipping. However, the alternative leads to saturating dv/v if there is a long term change.
twins = [[[40., 100.]], [[20., 50.]], [[-40., -16.],[-20., -8.], [8., 20.], [16., 40.]], [[-20., -8.], [-10., -4.], [4., 10.], [8., 20.]],
         [[-20., -8.], [-10., -4.], [4., 10.], [8., 20.]], [[-10., -4.], [-5., -2.], [4., 10.], [2., 5.]], [[2., 5.]], [[-5., -2.], [-2.5, -1.], [1., 2.5], [2., 5.]]]
badtimes = [[UTCDateTime("1998,180").timestamp, UTCDateTime("2000,001").timestamp], [UTCDateTime("2001,090").timestamp, UTCDateTime("2002,350").timestamp]]
freq_bands = [[0.1, 0.2], [0.2, 0.5], [0.5, 1.], [1., 2.], [1.25, 1.75], [2., 4.], [2.5, 3.5], [4., 8.]]

for iinf, input_file in enumerate(input_files):
    ixf = int(os.path.splitext(input_file)[0].split("_")[-1])
    if ixf < 2:
        continue
    if ixf >=3:
        ngrid=100
    station = os.path.basename(input_file.split(".")[1])
    ch1 = os.path.basename(input_file.split(".")[2][0: 3])
    ch2 = os.path.basename(input_file.split(".")[4])
    freq_band = freq_bands[ixf]
    ixf = int(os.path.basename(input_file).split(".")[-2][-1])
    dset = CCDataset(input_file)
    dset.data_to_memory()

    if rank == 0:    
        dset.interpolate_stacks(stacklevel=0, new_fs=new_fs)
        plot_output = re.sub("\.h5", "_nonnorm.png", os.path.basename(input_file))
        dset.plot_stacks(stacklevel=0, label_style="year",
                       seconds_to_show=plot_tmax[ixf], outfile=plot_output)
        print(dset.dataset[0].data.max())

    for twin in twins[ixf]:
        maxdvv = skipfactor * 1. / (2. * freq_band[1] * max(abs(np.array(twin))))
        print("maxdvv ", maxdvv)
        
        # copy
        if rank == 0:
            dset.dataset[1] = CCData(dset.dataset[0].data.copy(), dset.dataset[0].timestamps.copy(), dset.dataset[0].fs)
        # collect measured dvv and metadata in dictionary to save
        results = {}
        results["dvv_max"] = maxdvv
        results["input_data"] = os.path.basename(input_file)
        results["f0Hz"] = freq_band[0]
        results["f1Hz"] = freq_band[1]
        results["w0s"] = twin[0]
        results["w1s"] = twin[1]


        # window
        t_mid = (twin[0] + twin[1]) / 2.
        hw = (twin[1] - twin[0]) / 2.
        if rank == 0:
            dset.window_data(t_mid=t_mid, hw=hw, window_type="boxcar", stacklevel=1)
            # define the tasks of measuring dvv and allocate array
        else:
            n = 0

        if rank == 0:
            tstmps = dset.dataset[0].timestamps
            # cut out times where the station wasn't operating well
            bad_ixs = []
            for badwindow in badtimes:
                ixbw1 = np.argmin((tstmps - badwindow[0]) ** 2)
                ixbw2 = np.argmin((tstmps - badwindow[1]) ** 2)
                bad_ixs.extend(list(np.arange(ixbw1, ixbw2)))
            good_windows = [ixwin for ixwin in range(len(tstmps)) if not ixwin in bad_ixs]
            data = dset.dataset[1].data[good_windows]
            tstmps = dset.dataset[1].timestamps[good_windows]
            dset.dataset[2] = CCData(data, tstmps, dset.dataset[1].fs)
            n = len(dset.dataset[2].timestamps)
            k = int(n * (n - 1) / 2.)
            data_dvv = np.zeros(k)
            data_dvv_err = np.zeros(k)
                #for ixbad in range(ixbw1, ixbw2):
                #
                #    frac = (ixbw2 - ixbad) / (ixbw2 - ixbw1)
                #    data[ixbad, :] = frac * data[ixbw1, :] + (1-frac) * data[ixbw2, :] 

        # fill dvv array & G matrix
        n = comm.bcast(n, root=0)
        counter = 0
        for i in range(n):
            # i-th stack as reference
            if rank == 0:
                ref = dset.dataset[2].data[i, :]
            else:
                ref = None
            ref = comm.bcast(ref, root=0)
            dvv, dvv_timest, ccoeff, \
                best_ccoeff, dvv_error, cwtfreqs = dset.measure_dvv_par(f0=freq_band[0], f1=freq_band[1], ref=ref,
                                                                        ngrid=100, method="stretching",
                                                                        dvv_bound=maxdvv, stacklevel=2)
            if rank == 0:
                for j in range(i + 1, n):
                    data_dvv[counter] = dvv[j]
                    data_dvv_err[counter] = dvv_error[j]
                    counter += 1
        if rank == 0:
            print("data error median ", np.median(data_dvv_err))
            print("Measurement concluded ", time.strftime("H%H.M%M.S%S"))

            results["dvv_data"] = data_dvv
            results["dvv_err"] = data_dvv_err
            results["timestamps"] = tstmps #dset.dataset[1].timestamps
            np.save("alltoall_dvv_{}_{}_{}-{}_{}-{}Hz_{}-{}s.npy".format("stretching", station, ch1, ch2, *freq_band, *twin), results)
