from obspy import read

year = "1995"
stations = ["UNM"]
locations = ["", "00"]
channels = ["BHN", "BHE", "BHZ"]
timestrs = ["0*", "1*", "2*", "3*"]
outdir = "data/processed/merged/"

def name_processed_file(stats, startonly=False):

    inf = [
        stats.network,
        stats.station,
        stats.location,
        stats.channel
    ]

    t1 = stats.starttime.strftime('%Y.%j.%H.%M.%S')
    t2 = stats.endtime.strftime('%Y.%j.%H.%M.%S')
    if startonly:
        t2 = '*'

    inf.append(t1)
    inf.append(t2)

    inf.append(stats._format)

    filenew = '{}.{}.{}.{}.{}.{}.{}'.format(*inf)

    return filenew


for station in stations:
    for location in locations:
        for channel in channels:
            for time in timestrs:

                filestr = "data/processed/*" + station + "." + location + "." + channel + "." + year + "." + time
                try:
                    tr = read(filestr)
                except:
                    continue
                tr.merge(fill_value=0.0)
                outname = name_processed_file(tr[0].stats)
                tr.write(outdir + outname, format="MSEED")
                print(tr) 
