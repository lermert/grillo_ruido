from obspy.clients.fdsn import Client
from obspy import UTCDateTime

#######################################
# input:
c = Client("IRIS")
channel = 'HH*'
input_files = ["downloadlist.txt"]
t0 = UTCDateTime("2006-01-01T00:00:00")
t1 = UTCDateTime("2006-01-05T23:45:00")
twindow = 86400
only_response = False
#######################################

for input_file in input_files:
    f = open(input_file, 'r')
    stas = f.readlines()

    for s in stas:
        net = s.split('|')[0]
        sta = s.split('|')[1]
        if not only_response:
            # get data
            t = t0
            while t < t1:
                print(t)
                tr = c.get_waveforms(network=net, station=sta, location='*', channel=channel,
                                     starttime=t, endtime=t + twindow)
             
                # set filename
                output_file = "{}.{}.{}.MSEED".format(net, sta,
                                                  t0.strftime("%Y.%m.%dT%H.%M.%s"))
                # save data
                tr.write(output_file, format='MSEED')
                t += twindow
                print(output_file)

        # get inventory
        inv = c.get_stations(starttime=t0, endtime=t1, )
        # set filename
        fname = ("{}.{}.xml".format(net, sta))
        # save instrument response
        with open(fname, 'w') as fh:
            fh.write(inv)
