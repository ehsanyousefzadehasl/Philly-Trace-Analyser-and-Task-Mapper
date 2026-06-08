# Production Trace Characterization

This table summarizes successfully completed GPU jobs after each source trace is normalized. The supported subset contains jobs requesting one or two GPUs, matching the configured server-level evaluation scope.

| Trace | Cluster | Valid jobs | 1–2 GPU jobs | Job coverage | GPU-service coverage | 1-GPU fraction | 2-GPU fraction | Runtime p50 | Runtime p95 | Queue p50 | Queue p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Helios | Earth | 313,953 | 295,702 | 94.19% | 12.30% | 93.74% | 0.44% | 240 s | 13812 s | 0 s | 444 s |
| Philly | Philly | 83,140 | 79,126 | 95.17% | 21.55% | 92.96% | 2.21% | 1124 s | 39630 s | 178 s | 12586 s |
| Helios | Saturn | 406,982 | 289,626 | 71.16% | 6.39% | 68.84% | 2.33% | 190 s | 12652 s | 0 s | 1009 s |
| Helios | Uranus | 199,930 | 153,421 | 76.74% | 4.94% | 75.11% | 1.63% | 538 s | 35296 s | 0 s | 15189 s |
| Helios | Venus | 65,912 | 39,846 | 60.45% | 6.32% | 57.02% | 3.44% | 261 s | 56857 s | 0 s | 3989 s |

The original production traces remain separate from this repository. Values in this document are derived using the `trace-mapper summarize-source` command.
