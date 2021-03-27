from obspy import Stream, Trace, UTCDateTime
import json
from glob import glob
import os
import numpy as np
from warnings import warn
# input:
indir = "jsonl"  # could be set as command line input using sys.args
# handling overlaps:
interpolation_samples = 0
# 0: Do not interpolate
# n: interpolate n samples
# -1: interpolate ALL overlapping samples
misalignment_threshold = 0.5
# 0: do no align samples
# 0.5: align all samples with sub-sample time shift

# station renaming: SEED convention does not allow station names to be
# arbitrarily long. Also, human readable station names nice for some purposes
stamap = {"3ef3d787af85": "ANST", "094330d18e77": "BDTL",\
          "d11199910e51": "COAM", "029": "029",
          "8CAAB5A53D5C": "SEA1"}


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
station_previous = 'nostation'

for t in traces:
    print(t)
    f = open(t, 'r')
    f = f.readlines()
    f = ",".join(f)
    rec = json.loads(f)

    network = rec[0]["country_code"]
    if len(network) > 2:
        warn("Length of Country code used as network name may be too long for SEED")
    station = rec[0]["device_id"]
    print(network, station)

    # If this file belongs to another station than the previous: Save and start a new data stream
    if station != station_previous:
        if station_previous != 'nostation':
            s_out = Stream()
            for stream in [s_x, s_y, s_z]:
                s_out += clean_up(stream)
            s_out.write('{}.{}.MSEED'.format(network, station_previous),
                        format='MSEED')
        s_x = Stream()
        s_y = Stream()
        s_z = Stream()
        station_previous = station

    # add all the records in the file to one data stream
    for d in rec:
        if "sr" not in list(d.keys()):
            continue

        if "device_t" not in list(d.keys()):
            continue

        tr = Trace()
        tr.stats.sampling_rate = d['sr']
        tr.stats.starttime = UTCDateTime(d['device_t']) -\
            (len(d['x']) - 1) / d['sr']
        tr.stats.station = station
        if station in stamap.keys():
            tr.stats.station = stamap[station]
        else:
            warn("Station name for {} now {} to fit MSEED format".
                 format(station, station[0:4]))
            tr.stats.station = station[0: 4]
        tr.stats.network = network

        try:
            tr.data = np.array(d['x'])
            tr.stats.channel = 'ENX'
            s_x += tr.copy()
        except KeyError:
            pass

        try:
            tr.data = np.array(d['y'])
            tr.stats.channel = 'ENY'
            s_y += tr.copy()
        except KeyError:
            pass

        try:
            tr.data = np.array(d['z'])
            tr.stats.channel = 'ENZ'
            s_z += tr.copy()
        except KeyError:
            pass
s_out = Stream()
for stream in [s_x, s_y, s_z]:
    s_out += clean_up(stream)
print(s_out)


s_out.write('{}.{}.{}.{}.MSEED'.format(network, s_out[0].stats.station, 
                                       s_out[0].stats.starttime.strftime("%Y.%j"),
                                       s_out[-1].stats.endtime.strftime("%Y.%j")), format='MSEED')
