import pandas as pd
import matplotlib.pyplot as plt

# Path to your CSV file
csv_file = "philly_trace_90_tasks.csv"

# Load the CSV file
data = pd.read_csv(csv_file, header=None, names=["waiting_time", "tasks"])

# Extract the first column (waiting time)
waiting_times = data["waiting_time"]


print(waiting_times.sum())
# Plot a histogram of the waiting times
plt.figure(figsize=(10, 6))
plt.hist(waiting_times, bins=20, edgecolor='k', alpha=0.7)
plt.title("Histogram of Waiting Times")
plt.xlabel("Waiting Time (seconds)")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('waiting_time_histogram.png')