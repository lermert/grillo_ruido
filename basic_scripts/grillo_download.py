# coding: utf-8
import json
from obspy import UTCDateTime
from openeew.data.aws import AwsDataClient
import os

# output_folder
output_folder = 'jsonl'
# country: mx for Mexico
country = 'mx'
# date to check for devices
date_utc = '2020-01-01 00:00:00'
# start and endtime of window of interest
start_date_utc = '2020-01-01 00:00:00'
end_date_utc = '2020-01-05 00:00:00'
# client
data_client = AwsDataClient(country)  # mx = for Mexico
# either all available devices or a string list of device names
# devices = data_client.get_devices_as_of_date(date_utc)
devices = ['029']


# script:
os.makedirs(output_folder, exist_ok=True)
for dev_id in devices:
    records = data_client.get_filtered_records(start_date_utc, end_date_utc,
                                               device_ids=[dev_id])
    tst = UTCDateTime(int(records[0]["device_t"]))
    outfile = os.path.join(output_folder, country + "." + dev_id + "." +
                           tst.strftime("%Y.%jT%H-%M-%S") + '.jsonl')

    with open(outfile, 'w') as f:
        f.write(json.dumps(records))

# records_sorted = {}records_sorted = {}
# for rec in records:
#     if rec["device_id"] in list(records_sorted.keys()):
#         records_sorted[rec["device_id"]].append(rec)
#     else:
#         records_sorted[rec["device_id"]] = [rec]

# for k, record_list in records_sorted.items():
#     tst = UTCDateTime(int(record_list[0]["device_t"]))
#     outfile = os.path.join(output_folder, country + "." + k + "." +
#                            tst.strftime("%Y.%jT%H-%M-%S") + '.jsonl')
#     with open(outfile, 'w') as f:
#         f.write(json.dumps(record_list))
# for rec in records:
#     if rec["device_id"] in list(records_sorted.keys()):
#         records_sorted[rec["device_id"]].append(rec)
#     else:
#         records_sorted[rec["device_id"]] = [rec]

# for k, record_list in records_sorted.items():
#     tst = UTCDateTime(int(record_list[0]["device_t"]))
#     outfile = os.path.join(output_folder, country + "." + k + "." +
#                            tst.strftime("%Y.%jT%H-%M-%S") + '.jsonl')
#     with open(outfile, 'w') as f:
#         f.write(json.dumps(record_list))
