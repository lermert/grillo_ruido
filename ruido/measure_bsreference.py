from ruido.cc_dataset_mpi import CCDataset, CCData
from obspy import UTCDateTime
import os
import numpy as np
import time
from glob import glob
import re
from mpi4py import MPI
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

input_files = glob("UNM_BH_30d_raw_unaligned/G*.h5")
input_files.sort()
print(input_files)
measurement_type = "stretching"
align_mainlobe = True

new_fs = 160.0
plot_tmax = [100.0, 60.0, 40.0, 20.0, 20.0, 10.0, 10.0, 5.0]
skipfactor = 4  # allow this times more stretching as would be permitted to avoid cycle skipping
# note: this leads to cycle skipping. However, the alternative leads to saturating dv/v if there is a long term change.
twins = [[[40., 100.]], [[20., 50.]], [[-40., -16.],[-20., -8.], [8., 20.], [16., 40.]], [[-20., -8.], [-10., -4.], [4., 10.], [8., 20.]],
         [[-20., -8.], [-10., -4.], [4., 10.], [8., 20.]], [[-10., -4.], [-5., -2.], [4., 10.], [2., 5.]], [[2., 5.]], [[-5., -2.], [-2.5, -1.], [1., 2.5], [2., 5.]]]
freq_bands = [[0.1, 0.2], [0.2, 0.5], [0.5, 1.], [1., 2.], [1.25, 1.75], [2., 4.], [2.5, 3.5], [4., 8.]]

bootstrap_n = 50
ngrids = [50, 50, 50, 100, 100, 100, 100, 100]
ref_duration = 86400. * 365
for iinf, input_file in enumerate(input_files):
    ixf = int(os.path.splitext(input_file)[0].split("_")[-1])
    station = os.path.basename(input_file.split(".")[1])
    ch1 = os.path.basename(input_file.split(".")[2][0: 3])
    ch2 = os.path.basename(input_file.split(".")[4])
    freq_band = freq_bands[ixf]
    ixf = int(os.path.basename(input_file).split(".")[-2][-1])
    dset = CCDataset(input_file)
    dset.data_to_memory()
    ngrid = ngrids[ixf]

    if rank == 0:    
        dset.interpolate_stacks(stacklevel=0, new_fs=new_fs)

        plot_output = re.sub("\.h5", "_nonnorm.png", os.path.basename(input_file))
        dset.plot_stacks(stacklevel=0, label_style="year",
                       seconds_to_show=plot_tmax[ixf], outfile=plot_output)
        print(dset.dataset[0].data.max())

    for twin in twins[ixf]:
        maxdvv = skipfactor / (2. * freq_band[1] * max(abs(np.array(twin))))
        print(maxdvv)
        # copy
        if rank == 0:
            dset.dataset[1] = CCData(dset.dataset[0].data.copy(), dset.dataset[0].timestamps.copy(), dset.dataset[0].fs)

            if align_mainlobe:
                dset.dataset[1].align(-1./ (2. * freq_band[0]), 1. / (2. * freq_band[1]),
                                      ref=dset.dataset[1].data[dset.dataset[1].ntraces // 2, :])
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
            n = len(dset.dataset[1].timestamps)
            k = int(n * (n - 1) / 2.)
            data_dvv = np.zeros(k)
            data_dvv_err = np.zeros(k)
        else:
            n = 0

        if rank == 0:
            data = dset.dataset[1].data
            tstmps = dset.dataset[1].timestamps
            tstmps_bs = tstmps[tstmps < tstmps[-1] - ref_duration ]
        # fill dvv array & G matrix
        n = comm.bcast(n, root=0)
        counter = 0
        for i in range(bootstrap_n):
            if rank == 0:
                # choose a reference at random:
                ixref = np.random.choice(np.arange(len(tstmps_bs)))
                print("random t: ", UTCDateTime(tstmps_bs[ixref]))
                ws_ref = dset.group_for_stacking(t0=tstmps_bs[ixref], duration=ref_duration, stacklevel=1)
                dset.stack(ws_ref, stacklevel_in=1, stacklevel_out=2)
                ref = dset.dataset[2].data[-1, :]
            else:
                ref = None
            ref = comm.bcast(ref, root=0)
            dvv, dvv_timest, ccoeff, \
                best_ccoeff, dvv_error, cwtfreqs = dset.measure_dvv_par(f0=freq_band[0], f1=freq_band[1], ref=ref,
                                                                        ngrid=ngrid, method="stretching",
                                                                        dvv_bound=maxdvv, stacklevel=1)
            if rank == 0:

                results["dvv_data_{}".format(i)] = dvv
                results["dvv_err_{}".format(i)] = best_ccoeff
               # plt.plot(tstmps, dvv, linewidth=0.5)

        if rank == 0:
            results["timestamps"] = dset.dataset[1].timestamps
            np.save("bsref_dvv_{}_{}_{}-{}_{}-{}Hz_{}-{}s_{}.npy".format("stretching", station, ch1, ch2, *freq_band, *twin, maxdvv), results)

            # plt.show()
