@everywhere using SeisIO
@everywhere using Dates
@everywhere using Distributed

f = open("/home/lermert/code/grillo_ruido/data/downloadlist.txt", "r")
start_date = "2020-01-01T00:00:00"
end_date = "2020-01-03T00:00:00"
duration = 86400
channels = ["HNZ", "HNN", "HNE"]
outdir = "/home/lermert/code/grillo_ruido/data/"
out_format  = "mseed"
t0 = DateTime(start_date)
end_date = DateTime(end_date)
stations = split(read(f, String), "\n")
stations = [split(sta, "|") for sta in stations]

starttimes = []
let t_start=DateTime(start_date), t_step=Second(duration), t_end=DateTime(end_date)
  while t_start < t_end
    append!(starttimes, [t_start])
    t_start += t_step
  end
end


@everywhere function download_window_from_stations(starttime::DateTime,
                                       stations::Array{Array{SubString{String},1},1},
                                       channels::Array{String, 1},
                                       duration::Int64;
                                       out_format="mseed"::String,
                                       outdir="."::String)
  for sta in stations
      if length(sta) == 1
         continue
      end

      for cha in channels
        station_id = string(sta[1],".", sta[2],".00.", cha)
        println(station_id)


            if out_format == "mseed"
              stream = get_data("IRIS", station_id, s=starttime, t=duration, w=true, v=1)
            else
              stream = get_data("IRIS", station_id, s=starttime, t=duration, w=false, v=1)
            end

            if stream == nothing
                continue
            end

            if out_format == "native"
              wseis(stream, outdir * station_id * "." * string(start_date) * ".jdat")
            elseif out_format == "sac"
              writesac(stream)
              writesacpz(stream, outdir * station_id * ".pz")
            elseif out_format == "mseed"
              println("was written.")
            else
              print("Unknown output format")
              throw(ErrorException)
            end
      end
  end

end # function download_window_from_stations

pmap(starttime->download_window_from_stations(starttime, stations, channels, duration), starttimes)
