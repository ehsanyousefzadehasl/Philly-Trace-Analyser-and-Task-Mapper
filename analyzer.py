import json
import numpy as np
import pandas as pd
from datetime import datetime
date_format = '%Y-%m-%d %H:%M:%S'

# reading the Phily data
f = open('cluster_job_log')
data = json.load(f)

# the list for gathering the submitted time
submit_times = []

failed = 0
# loop for going over the instance and taking out their submit time
for i in data:
    # if len(i['attempts']) == 0:
    #     print(i['submitted_time'], "\n")

    #     submit_times.append(i['submitted_time'])

    #     continue
    # elif len(i['attempts'][-1]['detail']) == 0:
    #     print(i['submitted_time'], "\n")

    #     submit_times.append(i['submitted_time'])

    #     continue

    # print(i, "\n")
    # print(i['attempts'][-1]['detail'], "\n")
    # print(i['submitted_time'], "\n")

    print(i['status'])
    if i['status'] == 'Failed':
        print('jump over')
        failed += 1
        continue
        exit()
    date_obj = datetime.strptime(i['submitted_time'], date_format)

    # print(date_obj)

    submit_times.append(date_obj)
    # print(i['attempts'][-1]['detail'][-1]['gpus'], "\n")
    # exit()

print("failed: ", failed)

submit_times = np.array(submit_times)

df1 = pd.DataFrame(data, columns=['submitted_time'])

# print(df1)

df_sorted = df1.sort_values(by="submitted_time")

print(df_sorted)


# exit()
df_selected_day = df_sorted[(df_sorted["submitted_time"] > "2017-12-20") & (df_sorted["submitted_time"] < "2017-12-21")]

# print(df_selected_day)


from datetime import datetime
date_str = '2023-02-28 14:30:00'
date_format = '%Y-%m-%d %H:%M:%S'


# taking 100 of them
sampled_sequenced = df_selected_day.iloc[100:150, ]

import csv
f = open('philly_trace_100_tasks.csv', 'w')
writer = csv.writer(f)


previous = None
for idx, row in sampled_sequenced.iterrows():

    print(row)
    if previous == None:
        wait_time = 0
        previous = datetime.strptime(row[0], date_format)

        item = [wait_time, 1]
        
        writer.writerow(item)

    else:
        difference = datetime.strptime(row[0], date_format) - previous
        wait_time = difference.seconds

        previous = datetime.strptime(row[0], date_format)


        item = [wait_time, 1]

        writer.writerow(item)

        print(wait_time)

# print(sampled_sequenced)


# Sampling 100 of them to be able to do experiment.
# sampled_version = df_selected_day.sample(n=100, random_state=1)
# sampled_version = sampled_version.sort_values(by = "submitted_time")
# print(sampled_version)