import numpy as np
import pandas as pd
from obspy import UTCDateTime

def measurement_brenguier(dset, conf, twin, freq_band, rank, comm):
    if rank == 0:
        tstmps = dset.dataset[1].timestamps
        # cut out times where the station wasn't operating well
        # define in measurement_config.yml
        bad_ixs = []
        for badwindow in conf["badwins"]:
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
    else:
        n = 0

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
                                                                    ngrid=100, method=conf["mtype"],
                                                                    dvv_bound=maxdvv, stacklevel=2)
        comm.barrier()
        if rank == 0:
            for j in range(i + 1, n):
                data_dvv[counter] = dvv[j]
                data_dvv_err[counter] = dvv_error[j]
                counter += 1
        else:
            pass
    return(dvv_timest, dvv, ccoeff, best_ccoeff, dvv_error, None)

def measurement_bootstrap(dset, config, twin, freq_band, rank, comm):
    
    bootstrap_n = config["n_bootstrap"]
    ref_duration = config["r_duration"]

    t = []
    dvv = []
    cc0 = []
    cc1 = []
    err = []
    tags = []
    if rank == 0:
        tstmps_bs = dset.dataset[1].timestamps
        last_ix = np.argmin(np.abs(tstmps_bs - tstmps_bs[-1] + ref_duration))
        tstmps_bs = tstmps_bs[:last_ix]

    for i in range(bootstrap_n):
        print(dset)

        if rank == 0:
            # choose a reference at random:
            tref = np.random.choice(tstmps_bs)
            print("random t: ", UTCDateTime(tref))
            ws_ref = dset.group_for_stacking(t0=tref,
                                             duration=ref_duration,
                                             stacklevel=1)
            dset.stack(ws_ref, stacklevel_in=1, stacklevel_out=2)
            ref = dset.dataset[2].data[-1, :]
        else:
            ref = None
        ref = comm.bcast(ref, root=0)
        dvvp, dvv_timestp, ccoeffp, \
            best_ccoeffp, dvv_errorp, cwtfreqsp = dset.measure_dvv_par(f0=freq_band[0], f1=freq_band[1], ref=ref,
                                                                       ngrid=config["ngrid"],
                                                                       method=config["mtype"],
                                                                       dvv_bound=config["maxdvv"], stacklevel=1)
        comm.barrier()
        if rank == 0:
            t.extend(dvv_timestp)
            dvv.extend(dvvp[:, 0])
            cc0.extend(ccoeffp[:, 0])
            cc1.extend(best_ccoeffp[:, 0])
            err.extend(dvv_errorp[:, 0])
            tags.extend([i for j in range(len(dvv_timestp))])
        else:
            pass
    return(t, dvv, cc0, cc1, err, tags)


def measurement_reference(dset, config, twin, freq_band, rank, comm):
    
    t = []
    dvv = []
    cc0 = []
    cc1 = []
    err = []
    tags = []

    for i, reftimes in enumerate(config["r_windows"]):
        if rank == 0:
            tr0 = UTCDateTime(reftimes[0])
            tr1 = UTCDateTime(reftimes[1])
            ref_duration = tr1 - tr0
            ws_ref = dset.group_for_stacking(t0=tr0.timestamp,
                                             duration=ref_duration,
                                             stacklevel=1)
            dset.stack(ws_ref, stacklevel_in=1, stacklevel_out=2)
            ref = dset.dataset[2].data[-1, :]
        else:
            ref = None
        ref = comm.bcast(ref, root=0)
        dvvp, dvv_timestp, ccoeffp, \
            best_ccoeffp, dvv_errorp, cwtfreqsp = dset.measure_dvv_par(f0=freq_band[0], f1=freq_band[1], ref=ref,
                                                                       ngrid=config["ngrid"],
                                                                       method=config["mtype"],
                                                                       dvv_bound=config["maxdvv"], stacklevel=1)
        comm.barrier()
        if rank == 0:
            t.extend(dvv_timestp)
            dvv.extend(dvvp[:, 0])
            cc0.extend(ccoeffp[:, 0])
            cc1.extend(best_ccoeffp[:, 0])
            err.extend(dvv_errorp[:, 0])
            tags.extend([i for j in range(len(dvv_timestp))])
        else:
            pass
    return(t, dvv, cc0, cc1, err, tags)


def measurement_incremental(dset, config, twin, freq_band, rank, comm):
    
    t = []
    dvv = []
    cc0 = []
    cc1 = []
    err = []
    tags = []

    if rank == 0:
        ref = dset.dataset[1].data[0].copy()
        nd = len(dset.dataset[1].timestamps)
    else:
        ref = None
        nd = 0
    ref = comm.bcast(ref, root=0)
    nd = comm.bcast(nd, root=0)

    for i in range(nd):
        dvvp, dvv_timestp, ccoeffp, \
            best_ccoeffp, dvv_errorp, cwtfreqsp = dset.measure_dvv(f0=freq_band[0], f1=freq_band[1], ref=ref,
                                                                   ngrid=config["ngrid"],
                                                                   method=config["mtype"],
                                                                   dvv_bound=config["maxdvv"], stacklevel=1,
                                                                   indices=[i])
        comm.barrier()
        
        if rank == 0:
            t.extend(dvv_timestp)
            dvv.extend(dvvp[:, 0])
            cc0.extend(ccoeffp[:, 0])
            cc1.extend(best_ccoeffp[:, 0])
            err.extend(dvv_errorp[:, 0])
        else:
            pass

    return(t, dvv, cc0, cc1, err, tags)

def run_measurement(corrstacks, conf, twin, freq_band, rank, comm):
    output = pd.DataFrame(columns=["timestamps", "t0_s", "t1_s", "f0_Hz",  "f1_Hz",
                                   "tag", "dvv_max", "dvv", "cc_before", "cc_after",
                                   "dvv_err"])
    tags = None
    if conf["rtype"] == "inversion":
        t, dvv, cc0, cc1, err, tg = measurement_brenguier(corrstacks, conf, twin, freq_band, rank, comm)

    elif conf["rtype"] == "list":
        t, dvv, cc0, cc1, err, tags = measurement_reference(corrstacks, conf, twin, freq_band, rank, comm)

    elif conf["rtype"] == "bootstrap":
        t, dvv, cc0, cc1, err, tags = measurement_bootstrap(corrstacks, conf, twin, freq_band, rank, comm)
    
    elif conf["rtype"] == "increment":
        t, dvv, cc0, cc1, err, tg = measurement_incremental(corrstacks, conf, twin, freq_band, rank, comm)


    if rank == 0:
        for i in range(len(t)):
            if tags is None:
                tag = np.nan
            else:
                tag = tags[i]
            output.loc[i] = [t[i], twin[0], twin[1], freq_band[0], freq_band[1],
                             tag, conf["maxdvv"], dvv[i], cc0[i], cc1[i], err[i]]
        return(output)
    else:
        return(None)