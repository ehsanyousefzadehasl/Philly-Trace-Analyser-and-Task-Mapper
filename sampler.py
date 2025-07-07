import json
import pandas as pd
import csv
import argparse
from datetime import datetime

# Argument parser
parser = argparse.ArgumentParser(description="Sample submitted jobs from Philly trace")
parser.add_argument('--num_samples', type=int, default=90, help='Number of jobs to sample')
args = parser.parse_args()
num_samples = args.num_samples

# Load data
with open('cluster_job_log') as f:
    data = json.load(f)

# Filter out failed jobs
data = [job for job in data if job['status'] != 'Failed']
print(f"Failed jobs skipped: {len([j for j in data if j['status'] == 'Failed'])}")

# Convert to DataFrame and parse datetime
df = pd.DataFrame(data)
df['submitted_time'] = pd.to_datetime(df['submitted_time'])

# Sort by submission time
df_sorted = df.sort_values(by="submitted_time")

# Select jobs from a specific day
start, end = pd.Timestamp("2017-12-20"), pd.Timestamp("2017-12-21")
df_selected = df_sorted[(df_sorted["submitted_time"] > start) & (df_sorted["submitted_time"] < end)]

# Ensure enough samples are available
if len(df_selected) < num_samples:
    raise ValueError(f"Only {len(df_selected)} jobs found in the selected day. Requested {num_samples}.")

# Sample first N jobs (or change to .sample(...) for random sampling)
sampled = df_selected.iloc[:num_samples]

# Write wait times to CSV
with open('philly_trace_'+ str(num_samples) +'_tasks.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    previous = None
    for _, row in sampled.iterrows():
        if previous is None:
            wait_time = 0
        else:
            wait_time = int((row["submitted_time"] - previous).total_seconds())
        writer.writerow([wait_time, 1])
        previous = row["submitted_time"]