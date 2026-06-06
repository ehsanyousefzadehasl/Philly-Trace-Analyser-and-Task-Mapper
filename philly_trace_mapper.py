import csv
import random
import argparse

# Define the three command lists
# tasks less than 10 minutes
command_list1 = [
    "python src/submit.py --task workloads/efficientNet_cifar100_20e_2.rad\n",
    "python src/submit.py --task workloads/efficientNet_cifar100_20e_3.rad\n",

    "python src/submit.py --task workloads/resnet18_cifar100_20e_1.rad\n",
    "python src/submit.py --task workloads/resnet18_cifar100_20e_2.rad\n",
    "python src/submit.py --task workloads/resnet18_cifar100_20e_3.rad\n",

    "python src/submit.py --task workloads/resnet18_cifar100_50e_3.rad\n",

    "python src/submit.py --task workloads/resnet34_cifar100_20e_1.rad\n",
    "python src/submit.py --task workloads/resnet34_cifar100_20e_2.rad\n",
    "python src/submit.py --task workloads/resnet34_cifar100_20e_3.rad\n",

    "python src/submit.py --task workloads/mobilenet_cifar100_20e_2.rad\n",
    "python src/submit.py --task workloads/mobilenet_cifar100_20e_3.rad\n",
]

# tasks longer than 10 minutes and less than one hour
command_list2 = [
    "python src/submit.py --task workloads/efficientNet_cifar100_20e_1.rad\n",

    "python src/submit.py --task workloads/efficientNet_cifar100_50e_1.rad\n",
    "python src/submit.py --task workloads/efficientNet_cifar100_50e_2.rad\n",
    "python src/submit.py --task workloads/efficientNet_cifar100_50e_3.rad\n",

    "python src/submit.py --task workloads/resnet18_cifar100_50e_1.rad\n",
    "python src/submit.py --task workloads/resnet18_cifar100_50e_2.rad\n",
    
    "python src/submit.py --task workloads/resnet34_cifar100_50e_1.rad\n",
    "python src/submit.py --task workloads/resnet34_cifar100_50e_2.rad\n",
    "python src/submit.py --task workloads/resnet34_cifar100_50e_3.rad\n",

    "python src/submit.py --task workloads/mobilenet_cifar100_20e_1.rad\n",

    "python src/submit.py --task workloads/mobilenet_cifar100_50e_1.rad\n",
    "python src/submit.py --task workloads/mobilenet_cifar100_50e_2.rad\n",
    "python src/submit.py --task workloads/mobilenet_cifar100_50e_3.rad\n",

    "python src/submit.py --task workloads/Xception.rad\n",
    "python src/submit.py --task workloads/Xception2.rad\n",
    "python src/submit.py --task workloads/Xception3.rad\n",

    "python src/submit.py --task workloads/Inception.rad\n",
    "python src/submit.py --task workloads/Inception2.rad\n",
    "python src/submit.py --task workloads/Inception3.rad\n",

    "python src/submit.py --task workloads/vgg.rad\n",
    "python src/submit.py --task workloads/vgg2.rad\n",
    "python src/submit.py --task workloads/vgg3.rad\n",

    "python src/submit.py --task workloads/resnet.rad\n",
    "python src/submit.py --task workloads/resnet2.rad\n",
    "python src/submit.py --task workloads/resnet3.rad\n",

    "python src/submit.py --task workloads/mobilenet.rad\n",
    "python src/submit.py --task workloads/mobilenet2.rad\n",
    "python src/submit.py --task workloads/mobilenet3.rad\n",

    "python src/submit.py --task workloads/efficientNet.rad\n",
    "python src/submit.py --task workloads/efficientNet2.rad\n",
    "python src/submit.py --task workloads/efficientNet3.rad\n",

    "python src/submit.py --task workloads/mnist_train.rad\n",

    "python src/submit.py --task workloads/BERT_base.rad\n",
    "python src/submit.py --task workloads/BERT_large.rad\n",

    "python src/submit.py --task workloads/UNet.rad\n",

    "python src/submit.py --task workloads/dlrm.rad\n"
]

# tasks longer than one hour or with 2-GPU demand
command_list3 = [
    "python src/submit.py --task workloads/xlnet_base_cased_2.rad\n",
    "python src/submit.py --task workloads/xlnet_large_cased_2.rad\n",
    "python src/submit.py --task workloads/gpt2_large_2.rad\n",
    "python src/submit.py --task workloads/maskrcnn.rad\n"
]


# Argument parser for CSV input
parser = argparse.ArgumentParser(description="Generate a job submission scenario from Philly trace.")
parser.add_argument('--csv_file', type=str, required=True, help="Path to input CSV file (e.g., philly_trace_90_tasks.csv)")
args = parser.parse_args()


# Input and output files
input_csv = "philly_trace_60_tasks.csv"
output_script = f"philly_scenario_{args.csv_file}.sh"

# Weighted random selection using random.choices
def weighted_choice():
    # Choose between the two lists with 70% for command_list1 and 30% for command_list2
    chosen_list = random.choices(
        [command_list1, command_list2, command_list3],
        weights=[0.3, 0.6, 0.1],
        k=1
    )[0]
    return random.choice(chosen_list)

# Open the output script for writing
with open(output_script, "w") as f:
    # Open and read the CSV file
    with open(args.csv_file, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        
        for row in csv_reader:
            # Ensure the row has the expected number of columns
            if len(row) != 2:
                print(f"Skipping malformed row: {row}")
                continue
            
            # Extract sleep duration and number of tasks
            try:
                sleep_duration = int(row[0])  # Sleep duration
                task_count = int(row[1])     # Number of tasks
            except ValueError:
                print(f"Skipping row with invalid data: {row}")
                continue
            
            # Write sleep command to the script
            f.write(f"sleep {sleep_duration}\n")
            
            # Write task commands to the script
            for _ in range(task_count):
                f.write(weighted_choice())