import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Argument parser
parser = argparse.ArgumentParser(description="Plot histogram of job waiting times from a CSV trace file")
parser.add_argument('--csv_file', type=str, required=True, help='Path to the input CSV file (e.g., philly_trace_90_tasks.csv)')
args = parser.parse_args()

# Load the CSV file
data = pd.read_csv(args.csv_file, header=None, names=["waiting_time", "tasks"])

# Extract the first column (waiting time)
waiting_times = data["waiting_time"]

# Print total waiting time
print(f"Total waiting time: {waiting_times.sum()} seconds")

# Plot a histogram of the waiting times
plt.figure(figsize=(10, 6))
plt.hist(waiting_times, bins=20, edgecolor='k', alpha=0.7)
plt.title("Histogram of Waiting Times")
plt.xlabel("Waiting Time (seconds)")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save plot to file
output_file = 'waiting_time_histogram' + str(args.csv_file) + '.png'
plt.savefig(output_file)
print(f"Histogram saved to: {output_file}")