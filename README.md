# DL Trace Mapper

DL Trace Mapper generates reproducible deep-learning workload traces by mapping production GPU-cluster jobs to locally profiled executable workloads.

It supports Microsoft Philly traces, SenseTime Helios traces, workload-catalog construction, reproducible trace-window selection, GPU-demand-aware workload mapping, exclusive-execution estimation, and Markdown/figure reports.

## Installation

```bash
git clone https://github.com/ehsanyousefzadehasl/Philly-Trace-Analyser-and-Task-Mapper.git
cd Philly-Trace-Analyser-and-Task-Mapper
python -m pip install -e .
```

Verify the installation:

```bash
trace-mapper --help
```

## Quick Start

Build a workload catalog:

```bash
trace-mapper build-catalog \
  --profiles /path/to/solo_profile_results_1gpu.csv \
             /path/to/solo_profile_results_2gpu.csv \
  --task-root /path/to/project/root \
  --output outputs/workload_catalog.csv
```

Characterize all configured production traces:

```bash
trace-mapper summarize-suite \
  --config examples/configs/trace_suite.yaml
```

Generate a mapped trace:

```bash
trace-mapper generate \
  --config examples/configs/philly_source_preserving.yaml
```

Generate its report:

```bash
trace-mapper report-generation \
  --manifest outputs/generated_traces/philly_seed42_60jobs.manifest.json
```

## Documentation

- [Raw trace setup, provenance, licenses, and citations](data/README.md)
- [Production-trace characterization](docs/trace_characterization.md)
- [Generated-trace comparison](docs/generated_traces/generated_trace_comparison.md)

Example configurations:

- [Philly generation](examples/configs/philly_source_preserving.yaml)
- [Saturn generation](examples/configs/saturn_source_preserving.yaml)
- [Venus generation](examples/configs/venus_source_preserving.yaml)
- [Multi-trace characterization](examples/configs/trace_suite.yaml)

## Generated Artifacts

A generation run produces:

```text
<name>.csv                    # Enriched trace with provenance
<name>.execution.csv          # Minimal submit_time_s,task_path trace
<name>.manifest.json          # Reproducibility metadata and hashes
<name>.exclusive_jobs.csv     # Per-job exclusive simulation
<name>.exclusive_summary.json # Aggregate exclusive simulation
```

The execution CSV is scheduler-independent and contains only:

```text
submit_time_s,task_path
```

## Tests

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

## Scope

The included configurations target server-level experiments with 1-GPU and 2-GPU jobs. Larger GPU demands can be enabled when the workload catalog and target testbed support them. Exclusive-execution results are planning estimates, not measured scheduler outcomes.
