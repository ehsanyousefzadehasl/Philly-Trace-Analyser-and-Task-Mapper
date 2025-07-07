import matplotlib.pyplot as plt

# Function to simulate execution times and calculate metrics
def calculate_metrics(trace_file, task_execution_times_1, task_execution_time2, num_gpus=4):
    # Initialize GPU timelines
    gpu_timelines = [0] * num_gpus  # Tracks when each GPU will be free

    # Job metrics: (job_id, waiting_time, execution_time, completion_time)
    job_metrics = []
    current_time = 0  # Tracks the current simulation time (in minutes)

    with open(trace_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith("sleep"):
            # Sleep command: Update the current time (convert seconds to minutes)
            sleep_time = int(line.split()[1]) / 60
            current_time += sleep_time
        elif line.startswith("python src/submit.py"):
            # Extract task name and determine its execution time and GPU requirement
            task_name = line.split("--task workloads/")[1].strip()
            # print(task_name)
            if task_name in task_execution_times_1:
                task_time = task_execution_times_1[task_name]  # Task duration (in minutes)
                required_gpus = 1
            elif task_name in task_execution_time2:
                task_time = task_execution_time2[task_name]  # Task duration (in minutes)
                required_gpus = 2
            else:
                print(f"Error: Task '{task_name}' not found in execution times.")
                return

            # Allocate GPUs
            if required_gpus == 1:
                # Find the earliest available GPU
                gpu_id = gpu_timelines.index(min(gpu_timelines))
                start_time = max(current_time, gpu_timelines[gpu_id])  # Task starts after GPU is free
                completion_time = start_time + task_time  # When the task finishes
                gpu_timelines[gpu_id] = completion_time  # Update GPU availability
            else:  # For tasks requiring 2 GPUs
                # Find the two earliest available GPUs
                sorted_gpus = sorted(range(len(gpu_timelines)), key=lambda i: gpu_timelines[i])
                gpu1, gpu2 = sorted_gpus[:2]
                earliest_time = max(gpu_timelines[gpu1], gpu_timelines[gpu2])  # Both GPUs must be free
                start_time = max(current_time, earliest_time)  # Task starts after both GPUs are free
                completion_time = start_time + task_time  # When the task finishes
                gpu_timelines[gpu1] = completion_time  # Update GPU 1 availability
                gpu_timelines[gpu2] = completion_time  # Update GPU 2 availability

            # Store metrics for this job
            waiting_time = max(0, start_time - current_time)
            job_metrics.append((len(job_metrics) + 1, waiting_time, task_time, completion_time))

    # Calculate the end-to-end execution time
    end_to_end_time = max(gpu_timelines)

    return job_metrics, end_to_end_time


# Function to analyze and visualize job metrics
def analyze_and_plot_metrics(job_metrics, end_to_end_time):
    # Calculate averages
    total_waiting_time = sum(job[1] for job in job_metrics)
    total_execution_time = sum(job[2] for job in job_metrics)
    total_completion_time = sum(job[3] for job in job_metrics)
    num_jobs = len(job_metrics)

    avg_waiting_time = total_waiting_time / num_jobs
    avg_execution_time = total_execution_time / num_jobs
    avg_completion_time = total_completion_time / num_jobs

    print(f"Average Waiting Time: {avg_waiting_time:.2f} minutes")
    print(f"Average Execution Time: {avg_execution_time:.2f} minutes")
    print(f"Average Completion Time: {avg_completion_time:.2f} minutes")
    print(f"End-to-End Execution Time: {end_to_end_time:.2f} minutes")
    
    # Extract metrics for plotting
    job_ids = [job[0] for job in job_metrics]
    waiting_times = [job[1] for job in job_metrics]
    execution_times = [job[2] for job in job_metrics]

    # Plot metrics
    plt.figure(figsize=(12, 6))
    plt.bar(job_ids, waiting_times, label="Waiting Time", color="blue", alpha=0.6)
    plt.bar(job_ids, execution_times, bottom=waiting_times, label="Execution Time", color="green", alpha=0.6)
    plt.xlabel("Job ID")
    plt.ylabel("Time (minutes)")
    plt.title("Job Timeline Metrics (Waiting and Execution Time)")
    plt.legend()
    plt.tight_layout()
    plt.savefig('job_metrics.png')
    plt.show()

    return avg_waiting_time, avg_execution_time, avg_completion_time


# Example execution times for tasks using 1 GPU
task_execution_times_1 = {
    "Xception.rad": 46.86,
    "Xception2.rad": 45.77,
    "Xception3.rad": 44.44,
    "Inception.rad": 46.86,
    "Inception2.rad": 45.77,
    "Inception3.rad": 44.44,
    "vgg.rad": 48.44,
    "vgg2.rad": 44.38,
    "vgg3.rad": 42.41,
    "resnet.rad": 36.31,
    "resnet2.rad": 35.5,
    "resnet3.rad": 35,
    "mobilenet.rad": 36.01,
    "mobilenet2.rad": 35.42,
    "mobilenet3.rad": 34.91,
    "efficientNet.rad": 36.21,
    "efficientNet2.rad": 35.41,
    "efficientNet3.rad": 35.21,
    "mnist_train.rad": 2,
    "BERT_base.rad": 14.86,
    "BERT_large.rad": 44.93,
    "gpt2_xl.rad": 30.68,

    "efficientNet_cifar100_20e_1.rad": 15.332, 
    "efficientNet_cifar100_20e_2.rad":9.6,
    "efficientNet_cifar100_20e_3.rad":5.468,
    "efficientNet_cifar100_50e_1.rad":38.33,
    "efficientNet_cifar100_50e_2.rad":24,
    "efficientNet_cifar100_50e_3.rad":13.67,

    "mobilenet_cifar100_20e_1.rad":10.72577326,
    "mobilenet_cifar100_20e_2.rad":6.441957235,
    "mobilenet_cifar100_20e_3.rad":4.353343333,
    "mobilenet_cifar100_50e_1.rad":26.81443314,
    "mobilenet_cifar100_50e_2.rad":16.10489309,
    "mobilenet_cifar100_50e_3.rad":10.88335833,

    "resnet18_cifar100_20e_1.rad":6.568,
    "resnet18_cifar100_20e_2.rad":4.4,
    "resnet18_cifar100_20e_3.rad":3.268,
    "resnet18_cifar100_50e_1.rad":16.42,
    "resnet18_cifar100_50e_2.rad":11,
    "resnet18_cifar100_50e_3.rad":8.17,

    "resnet34_cifar100_20e_1.rad":9.736333333,
    "resnet34_cifar100_20e_2.rad":6.054628372,
    "resnet34_cifar100_20e_3.rad":4.018158833,
    "resnet34_cifar100_50e_1.rad":24.34083333,
    "resnet34_cifar100_50e_2.rad":15.13657093,
    "resnet34_cifar100_50e_3.rad":10.04539708,
}

# Example execution times for tasks using 2 GPUs (in minutes)
task_execution_time2 = {
    "xlnet_base_cased_2.rad": 71.59,
    "xlnet_large_cased_2.rad": 75.94,
    "gpt2_large_2.rad": 64.96,
}

# Path to the trace file
trace_file = "philly_scenario.sh"

# Calculate job metrics and end-to-end execution time
job_metrics, end_to_end_time = calculate_metrics(trace_file, task_execution_times_1, task_execution_time2, num_gpus=4)

# Analyze and plot metrics
if job_metrics:
    analyze_and_plot_metrics(job_metrics, end_to_end_time)