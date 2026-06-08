# DL Trace Mapper

DL Trace Mapper generates reproducible deep-learning workload traces by mapping production GPU-cluster jobs to locally profiled executable workloads.

It supports Microsoft Philly traces, SenseTime Helios traces, workload-catalog construction, reproducible trace-window selection, GPU-demand-aware workload mapping, exclusive-execution estimation, and Markdown/figure reports.

## Installation

```bash
git clone https://github.com/ehsanyousefzadehasl/dl-trace-mapper.git
cd dl-trace-mapper
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


## One-Command Workflow

Copy the public pipeline template to a machine-specific local configuration:

```bash
cp \
  examples/configs/full_pipeline.example.yaml \
  examples/configs/full_pipeline.local.yaml
```

Edit `examples/configs/full_pipeline.local.yaml` and set:

- the one-GPU and two-GPU solo-profile CSV paths;
- the workload project root;
- any machine-specific input paths.

Then run the complete workflow:

```bash
trace-mapper run-pipeline \
  --config examples/configs/full_pipeline.local.yaml
```

The pipeline:

1. builds the workload catalog from solo profiling results;
2. characterizes the configured production traces;
3. selects representative Philly, Saturn, and Venus windows;
4. maps source jobs to executable workloads;
5. generates per-trace and cross-trace reports;
6. creates fidelity figures and machine-readable CSV artifacts;
7. validates job count, runtime-CDF fidelity, GPU-demand fidelity, and 2-GPU workload coverage.

The local pipeline configuration is intentionally ignored by Git because it may contain machine-specific absolute paths. Commit only `examples/configs/full_pipeline.example.yaml`.

## Trace Evaluation Evidence

The generated documentation exposes the complete evidence chain used to validate the traces:

- [Production trace characterization](docs/trace_characterization.md)
- [Generated trace comparison](docs/generated_traces/generated_trace_comparison.md)
- [Representative trace fidelity](docs/generated_traces/representative_fidelity.md)
- [Pipeline validation summary](docs/generated_traces/pipeline_summary.json)
- [Raw trace sources and citations](data/README.md)

The fidelity report compares each full eligible source population with its selected 60-job window and mapped executable trace. It reports GPU-demand proportions, runtime-CDF buckets, interarrival behavior, representativeness scores, and suite-level coverage of the available 2-GPU workloads.
