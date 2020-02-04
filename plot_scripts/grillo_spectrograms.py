from obspy import read, UTCDateTime
import os
from matplotlib.pyplot import cm

# some input
files = ['mseed/mx.004.MSEED',
         'mseed/mx.001.MSEED']
freq_min = 0.05  # 0.1
freq_max = 1.0  # 10.0
corners = 10
zerophase = False
window_length_for_spectrogram = 1200  # seconds
window_length_for_fft = 30  # seconds
minimum_length = 0  # seconds
mult = 2  # padding for spectrogram
olap = 0.75
# clips of plot colorscale
lower_clip = 0  # 0.25
upper_clip = 1  # 0.75
log_freq = True
t0 = UTCDateTime("2020-01-28T19:10:00")
t1 = UTCDateTime("2020-01-28T20:00:00")


def apply_filter(trace):
    for windowed_tr in trace.slide(window_length=3600.0, step=1800.0):
        windowed_tr.detrend('demean')
        windowed_tr.detrend('linear')
    trace.taper(0.02)
    trace.filter('bandpass', freqmin=freq_min, freqmax=freq_max,
                 corners=corners, zerophase=zerophase)


for file in files:
    tr = read(file)
    tr = tr.trim(starttime=t0, endtime=t1)
    apply_filter(tr)
    print(tr)
    for component in ['ENX', 'ENY', 'ENZ']:
        tr_p = tr.select(channel=component)

        for tr_pp in tr_p.slide(window_length=window_length_for_spectrogram,
                                step=window_length_for_spectrogram,
                                include_partial_windows=False):
            if len(tr_pp) == 1:
                tr_pp = tr_pp[0]
            else:
                tr_pp = tr_pp.merge(fill_value=0)[0]
            if tr_pp.stats.delta * tr_pp.stats.npts < minimum_length:
                continue
            outfile = os.path.splitext(file)[0]
            of = outfile + '.' +\
                 component + '.' +\
                 tr_pp.stats.starttime.strftime("%Y.%m.%dT%H:%M") + '.spec.png'
            tr_pp.spectrogram(outfile=of,
                              cmap=cm.viridis,
                              wlen=window_length_for_fft, mult=mult,
                              dbscale=False,
                              clip=[lower_clip, upper_clip],
                              per_lap=olap,
                              log=log_freq)
