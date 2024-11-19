import csv
import random

# Define the two command lists
command_list1 = [
    "python submit.py --task Xception.rad\n",
    "python submit.py --task Xception2.rad\n",
    "python submit.py --task Xception3.rad\n",

    "python submit.py --task vgg.rad\n",
    "python submit.py --task vgg2.rad\n",
    "python submit.py --task vgg3.rad\n",

    "python submit.py --task resnet.rad\n",
    "python submit.py --task resnet2.rad\n",
    "python submit.py --task resnet3.rad\n",

    "python submit.py --task mobilenet.rad\n",
    "python submit.py --task mobilenet2.rad\n",
    "python submit.py --task mobilenet3.rad\n",

    "python submit.py --task efficientNet.rad\n",
    "python submit.py --task efficientNet2.rad\n",
    "python submit.py --task efficientNet3.rad\n",

    "python submit.py --task mnist_train.rad\n",

    "python submit.py --task BERT_base.rad\n",
    "python submit.py --task BERT_large.rad\n",

    "python submit.py --task gpt2_xl.rad\n",
]

command_list2 = [
    "python submit.py --task xlnet_base_cased_2.rad\n",
    "python submit.py --task xlnet_large_cased_2.rad\n",
    "python submit.py --task gpt2_large_2.rad\n",
]

# Input and output files
input_csv = "philly_trace_100_tasks.csv"
output_script = "philly_scenario.sh"

# Weighted random selection using random.choices
def weighted_choice():
    # Choose between the two lists with 70% for command_list1 and 30% for command_list2
    chosen_list = random.choices(
        [command_list1, command_list2],
        weights=[0.7, 0.3],
        k=1
    )[0]
    return random.choice(chosen_list)

# Open the output script for writing
with open(output_script, "w") as f:
    # Open and read the CSV file
    with open(input_csv, "r") as csv_file:
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