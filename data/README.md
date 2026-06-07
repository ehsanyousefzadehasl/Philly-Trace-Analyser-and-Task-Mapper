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
