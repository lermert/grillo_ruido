from ruido.cc_dataset_mpi import CCDataset
from obspy import UTCDateTime
from cmcrameri import cm
import matplotlib.pyplot as plt
from glob import glob

dsets = glob("stacks/*bp3*.h5")
# dsets = ["stacks/G.UNM.BHN--G.UNM.BHE.ccc.stacks_10.0days1995-2021_bp3_cl1.h5",
#          "stacks/G.UNM.BHZ--G.UNM.BHE.ccc.stacks_10.0days1995-2021_bp3_cl2.h5",
#          "stacks/G.UNM.BHZ--G.UNM.BHN.ccc.stacks_10.0days1995-2021_bp3_cl2.h5"]
bp_to_lag = {"0": 50, "1": 40, "2": 20, "3": 10, "4": 6}
bp_to_freqband = {"0": "0.375Hz", "1": "0.75Hz", "2": "1.5Hz", "3": "3.0Hz", "4": "6.0Hz"}
step = 10. * 86400.

for f in dsets:
    dset = CCDataset(f)
    dset.data_to_memory()
    channel1 = f.split(".")[2][0:3]
    channel2 = f.split(".")[4][0:3]
    bp = f.split("_")[-2][-1]
    freqband = bp_to_freqband[bp]
    lag = bp_to_lag[bp]
    cl = f.split("_")[-1][2]

    # dset.interpolate_stacks(new_fs=160, stacklevel=0)
    # fig = plt.figure(figsize=(8, 4.5))
    # ax = fig.add_subplot("111")
    # dset.plot_stacks(stacklevel=0, plot_mode="heatmap", scale_factor_plotting=0.5, cmap=cm.roma, label_style="year",
    #                  seconds_to_start=-lag, seconds_to_show=lag, normalize_all=1, ax=ax)
    # plt.savefig("stacks_{}{}_{}_{}cl_roma.png".format(channel1, channel2, freqband, cl))
    # plt.close()

    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot("111")
    dset.plot_stacks(stacklevel=0, plot_mode="heatmap", scale_factor_plotting=0.5, cmap=cm.broc, label_style="year",
                     seconds_to_start=-lag, seconds_to_show=lag, normalize_all=1, ax=ax, mask_gaps=True, step=step)
    plt.tight_layout()
    plt.savefig("stacks_{}{}_{}_{}cl_broc.png".format(channel1, channel2, freqband, cl))
    # fig = plt.figure(figsize=(8, 4.5))
    # ax = fig.add_subplot("111")
    # dset.plot_stacks(stacklevel=0, plot_mode="heatmap", scale_factor_plotting=0.5, cmap=plt.cm.jet, label_style="year",
    #                  seconds_to_start=-lag, seconds_to_show=lag, normalize_all=1, ax=ax)
    # plt.savefig("stacks_{}{}_{}_{}cl_jet.png".format(channel1, channel2, freqband, cl))
    # plt.close()

    del dset
