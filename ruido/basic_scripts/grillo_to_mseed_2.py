from obspy import Stream, Trace, UTCDateTime
import json
from glob import glob
import os
import numpy as np
import time
# input:
indir = 'jsonl/'
# handling overlaps:
interpolation_samples = 2
# 0: Do not interpolate, n: interpolate n samples
# -1: interpolate all overlapping samples
misalignment_threshold = 0.5
# minimum length of segments in seconds
min_len_segment = 1
# 0: do no align samples, 0.5: align all samples with sub-sample shift
stamap = {"3ef3d787af85": "ANST", "094330d18e77": "BDTL",\
          "d11199910e51": "COAM", "029": "029"}


def clean_up(stream_in):
    t_in = time.time()
    # align subsample shifts
    stream_in.merge(method=-1,
                    misalignment_threshold=misalignment_threshold)

    # close overlaps
    # (either discarding or interpolating the overlapping samples)
    #stream_in.merge(method=1, fill_value=None,
    #                interpolation_samples=interpolation_samples)
    # s = stream_in.split()  # do not return a masked array
    stream_in.taper(0.1)
    stream_in.merge(method=1, fill_value=0,
                    interpolation_samples=interpolation_samples)
    print("Conversion took {} seconds".format(time.time()-t_in))
    return(stream_in)


traces = glob(os.path.join(indir, '*jsonl'))
traces.sort()
station_previous = ''
for t in traces:
    print(t)
    network, station = os.path.basename(t).split('.')[0: 2]

    if station != station_previous:
        if station_previous != '':
            s_out = Stream()
            for stream in [s_x, s_y, s_z]:
                s_out += clean_up(stream)
            if len(s_out) > 0:
                s_out.write('{}.{}.{}.MSEED'.format(network, stamap[station_previous],
                            s_out[0].stats.starttime.strftime("%Y.%jT%H.%M.%S")),
                            format='MSEED')
        s_x = Stream()
        s_y = Stream()
        s_z = Stream()
        station_previous = station
    f = open(t, 'r')
    f = json.loads(f.read())

    for d in f:
        tr = Trace()
        tr.stats.sampling_rate = d['sr']
        tr.stats.starttime = UTCDateTime(d['device_t']) -\
            (len(d['x']) - 1) / d['sr']
        tr.stats.station = stamap[station]
        tr.stats.network = network

        if "previous_start" in locals():
            if tr.stats.starttime.strftime("%j.%H") != previous_start:
                s_out = Stream()
                for stream in [s_x, s_y, s_z]:
                    s_out += clean_up(stream)
                if len(s_out) > 0:
                    s_out.write('{}.{}.{}.MSEED'.format(network, stamap[station_previous],
                                s_out[0].stats.starttime.strftime("%Y.%jT%H.%M.%S")),
                                format='MSEED')
                s_x = Stream()
                s_y = Stream()
                s_z = Stream()
        else:
            pass
        previous_start = tr.stats.starttime.strftime("%j.%H")
        tr.data = np.array(d['z'])
        tr.data -= np.mean(tr.data)
        tr.stats.channel = 'ENX'
        s_x += tr.copy()
        tr.data = np.array(d['y'])
        tr.data -= np.mean(tr.data)
        tr.stats.channel = 'ENY'
        s_y += tr.copy()
        tr.data = np.array(d['x'])
        tr.data -= np.mean(tr.data)
        tr.stats.channel = 'ENZ'
        s_z += tr.copy()

s_out = Stream()
for stream in [s_x, s_y, s_z]:
    s_out += clean_up(stream)
for tr in s_out:
    print("max accel (unfiltered): ", tr.data.max())
if len(s_out) > 0:
    s_out.write('{}.{}.{}.MSEED'.format(network, stamap[station],
                s_out[0].stats.starttime.strftime("%Y.%jT%H.%M.%S")),
                format='MSEED')
