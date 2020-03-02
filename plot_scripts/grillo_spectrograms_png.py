from obspy import read, UTCDateTime
import os
import matplotlib.pyplot as plt
from glob import glob

# Input
# All files in one list will be plotted together
files_list = glob('/home/ubuntu/data_mount/ruido/panama_quake/mx.029*.MSEED')
files_list.extend(glob('/home/ubuntu/data_mount/ruido/panama_quake/G.UNM.MSEED'))

freq_min = 0.05  # 0.1
freq_max = 3.0  # 10.0
corners = 10
zerophase = False
# time: starttime and duration of spectrogram
t0 = UTCDateTime("2020-01-28T19:12:00")
window_length_for_spectrogram = 1200  # seconds
# each window length
window_length_for_fft = 30  # seconds
mult = 2  # padding for spectrogram
olap = 0.75
# clips of plot colorscale (1 = clip at maximum, 0.5 = clip at 50% maximum)
lower_clip = 0  # 0.25
upper_clip = 1  # 0.75
#  logarithmic frequency axis
log_freq = True
# figure size
fgsz = (16, 9)

def apply_filter(trace):
    for windowed_tr in trace.slide(window_length=3600.0, step=1800.0):
        windowed_tr.detrend('demean')
        windowed_tr.detrend('linear')
    trace.taper(0.02)
    trace.filter('bandpass', freqmin=freq_min, freqmax=freq_max,
                 corners=corners, zerophase=zerophase)



fig, axarray = plt.subplots(nrows=len(files_list), ncols=3, figsize=fgsz,
                                sharey='all')

sta = os.path.basename(files_list[0]).split('.')[1]
net = os.path.basename(files_list[0]).split('.')[0]
flnmstr = net
for i in range(len(files_list)):
    flnmstr += '.' + os.path.basename(files_list[i]).split('.')[1]
    tr = read(files_list[i])
    tr = tr.trim(starttime=t0, endtime=t0 + window_length_for_spectrogram)
    apply_filter(tr)
    tr.merge(fill_value=0)

    for j, tr_p in enumerate(tr):
        ax = axarray[i, j]
        tr_p.spectrogram(cmap=plt.cm.viridis,
                             wlen=window_length_for_fft, mult=mult,
                             dbscale=False,
                             clip=[lower_clip, upper_clip],
                             per_lap=olap,
                             log=log_freq,
                             axes=ax)
        if i == len(files_list) - 1:
            ax.set_xlabel('Time (s) after starttime')
        if j == 0:
            ax.set_ylabel('Frequency (Hz)')
        ax.set_ylim([freq_min, freq_max])
        ax.set_title(tr_p.id)
plt.tight_layout()
figname = '{}.{}.png'.format(flnmstr, t0.strftime("%Y.%m.%dT%H-%M-%s"))
fig.savefig(figname, dpi=150)
