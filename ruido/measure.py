from ruido.cc_dataset_mpi import CCDataset, CCData
from ruido.measurements import run_measurement
from obspy import UTCDateTime
import os
import numpy as np
import pandas as pd
import time
from glob import glob
import re
import yaml
import io
import sys
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

configfile = sys.argv[1]

# get configuration
config = yaml.safe_load(open(configfile))

# read in the input ==========================================================
input_files = glob(config["input_files"])
conf = {}
conf["mtype"] = config["measurement_type"]
conf["rtype"] = config["reference_type"]
new_fs = config["new_fs"]
conf["ngrid"] = int(config["ngrid"])
plot_tmax = config["plot_tmax"]
window_type = config["window_type"]
skipfactor = 4
# frequency bands
freq_bands = config["freq_bands"]
# time windows associated with each frequency band
twins = config["twins"]
assert len(freq_bands) == len(twins), "Number of time window \
lists must match number of frequency bands"
for fp in freq_bands:
    assert len(fp) == 2, "Frequency bands must be specified as \
tuples of fmin, fmax"
# reference type specific configs
# "Bad times" to be suppressed for measurement in inv. (Brenguier et al)
if config["reference_type"] == "inversion":
    conf["badwins"] = [[UTCDateTime(bt) for bt in entr] \
                       for entr in config["skiptimes_inversion"]]
if config["reference_type"] == "bootstrap":
    conf["r_duration"] = config["reference_length_days"] * 86400.
    conf["n_bootstrap"] = config["bootstrap_samples"]
if config["reference_type"] == "list":
    conf["r_windows"] = [[UTCDateTime(reft) for reft in entr]\
                         for entr in config["reference_list"]]

input_files.sort()
# ====================================================================

# For each input file:
# Read in the stacks
# For each time window:
# measurement
# save the result

for iinf, input_file in enumerate(input_files):
    ixf = int(os.path.splitext(input_file)[0].split("_")[-1])
    station = os.path.basename(input_file.split(".")[1])
    ch1 = os.path.basename(input_file.split(".")[2][0: 3])
    ch2 = os.path.basename(input_file.split(".")[4])

    freq_band = freq_bands[ixf]
    ixf = int(os.path.basename(input_file).split(".")[-2][-1])

    # read into memory
    dset = CCDataset(input_file)
    dset.data_to_memory()

    # interpolate and plot the stacks
    if rank == 0:
        if dset.dataset[0].fs != new_fs:
            dset.interpolate_stacks(stacklevel=0, new_fs=new_fs)
        plot_output = re.sub("\.h5", "_{}.png".format(ixf),
                             os.path.basename(input_file))
        dset.plot_stacks(stacklevel=0, label_style="year",
                       seconds_to_show=plot_tmax[ixf], outfile=plot_output)
        print(dset.dataset[0].data.max())

    # set up the dataframe to collect the results
    output = pd.DataFrame(columns=["timestamps", "t0_s", "t1_s", "f0_Hz",  "f1_Hz",
                                   "tag", "dvv_max", "dvv", "cc_before", "cc_after",
                                   "dvv_err"])
    # find max. dvv that will just be short of a cycle skip
    # then extend by skipfactor
    for twin in twins[ixf]:
        maxdvv = skipfactor * 1. / (2. * freq_band[1] *\
            max(abs(np.array(twin))))
        conf["maxdvv"] = maxdvv
        # print("maxdvv ", maxdvv)
        # window
        t_mid = (twin[0] + twin[1]) / 2.
        hw = (twin[1] - twin[0]) / 2.
        if rank == 0:
            dset.dataset[1] = CCData(dset.dataset[0].data.copy(),
                                     dset.dataset[0].timestamps.copy(),
                                     dset.dataset[0].fs)
            dset.window_data(t_mid=t_mid, hw=hw, window_type=window_type, stacklevel=1)
        else:
            pass

        print(dset)

        output_table = run_measurement(dset, conf, twin, freq_band, rank, comm)
        
        output = pd.concat([output, output_table], ignore_index=True)

comm.barrier()
# at the end write all to file
if rank == 0:
    outfile_name = "{}_{}{}_{}_{}.csv".format(station, ch1, ch2, conf["mtype"],
                                              conf["rtype"])
    output.to_csv(outfile_name)
