from obspy import read
from obspy.io.sac.sacpz import attach_paz
from obspy.signal import PPSD
from glob import glob
import os

data = glob("data/unam_accel/*.SAC")
tr0 = read(data[0])[0]

[net, sta, loc, cha] = os.path.basename(data[0]).split('.')[6: 10]
print(net, sta, loc, cha)
attach_paz(tr0, paz_file="data/meta/sacpz/" + "{}.{}.{}.{}.sacpz".format(net, sta, loc, cha))

# 
paz = {}
for k, v in tr0.stats.paz.items():
    paz[k] = v

p = PPSD(tr0.stats, metadata=paz, special_handling='ringlaser', period_limits=[0.1, 10.0])

for file in data:
    p.add(read(file))
p.plot()
