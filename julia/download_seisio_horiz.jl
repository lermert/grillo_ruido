using SeisIO
using Dates

f = open("/home/ubuntu/data_mount/ruido/unam_accelerometer/downloadlist.txt", "r")
start_date = "2013-02-15T00:00:00"
end_date = "2020-01-01T00:00:00"
t = 86400
channels = ["HNE", "HNN"]

t0 = DateTime(start_date)
end_date = DateTime(end_date)
stations = split(read(f, String), "\n")
stations = [split(sta, "|") for sta in stations]

for sta in stations
    if length(sta) == 1
       continue
    end
    for cha in channels
      station_id = string(sta[1],".", sta[2],".00.", cha)
      println(station_id)
      let t_start=DateTime(start_date), t_step=Second(t), t_end=DateTime(end_date)
        while t_start < t_end
          stream = get_data("IRIS", station_id, s=t_start, t=t, w=false, v=1)
          if stream == nothing
              continue
          end
          wseis("/home/ubuntu/data_mount/ruido/unam_accelerometer/" * station_id * "." * string(t_start) * ".mseed", stream)
          t_start += t_step
        end
      end
    end
end
