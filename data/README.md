# Trace Data

Raw production traces are not bundled with this repository.

## Microsoft Philly Trace

Download the public Philly trace from the official Microsoft Research
`msr-fiddle/philly-traces` repository and extract `cluster_job_log` to:

    data/raw/philly/cluster_job_log

The Philly trace is distributed under the Creative Commons Attribution 4.0
license. Users should cite:

Jeon et al., "Analysis of Large-Scale Multi-Tenant GPU Clusters for DNN
Training Workloads," USENIX ATC 2019.

Generated and trimmed trace files may be stored under:

    examples/source_traces/
    examples/generated_traces/

## SenseTime Helios Traces

The raw Helios production traces are not bundled with this repository.

Download them from the official HeliosData repository:

    https://github.com/S-Lab-System-Group/HeliosData

The public dataset contains traces from four GPU clusters:

- Earth
- Saturn
- Uranus
- Venus

Place the downloaded files using this directory structure:

    data/raw/helios/earth/cluster_log.csv
    data/raw/helios/saturn/cluster_log.csv
    data/raw/helios/uranus/cluster_log.csv
    data/raw/helios/venus/cluster_log.csv

Optional cluster-capacity files may be placed beside each trace as:

    data/raw/helios/<cluster>/cluster_gpu_number.csv

The Helios dataset is maintained separately from this project and remains
subject to its original license and attribution requirements.

When using the Helios traces, cite:

Qinghao Hu et al., "Characterization and Prediction of Deep Learning
Workloads in Large-Scale GPU Datacenters," SC 2021.
DOI: 10.1145/3458817.3476223.
