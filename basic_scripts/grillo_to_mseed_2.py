from obspy import Stream, Trace, UTCDateTime
import json
from glob import glob
import os
import numpy as np

# input:
indir = 'jsonl/'
# handling overlaps:
interpolation_samples = 0
# 0: Do not interpolate, n: interpolate n samples
# -1: interpolate all overlapping samples
misalignment_threshold = 0.5
# 0: do no align samples, 0.5: align all samples with sub-sample shift


def clean_up(stream_in):
    # align subsample shifts
    stream_in.merge(method=-1,
                    misalignment_threshold=misalignment_threshold)
    # close overlaps
    # (either discarding or interpolating the overlapping samples)
    stream_in.merge(method=1, fill_value=None,
                    interpolation_samples=interpolation_samples)
    s = stream_in.split()  # do not return a masked array
    return(s)


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
            s_out.write('{}.{}.MSEED'.format(network, station_previous),
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
        tr.stats.station = station
        tr.stats.network = network

        tr.data = np.array(d['x'])
        tr.stats.channel = 'ENX'
        s_x += tr.copy()
        tr.data = np.array(d['y'])
        tr.stats.channel = 'ENY'
        s_y += tr.copy()
        tr.data = np.array(d['z'])
        tr.stats.channel = 'ENZ'
        s_z += tr.copy()

s_out = Stream()
for stream in [s_x, s_y, s_z]:
    s_out += clean_up(stream)

s_out.write('{}.{}.MSEED'.format(network, station), format='MSEED')
