# DeepSeek-V4.1-Flash-1017 aggregate measurement archive

This archive contains allowlisted aggregate measurements, exact framework-entrypoint
runtime facts, terminal/source/cleanup proof summaries and the scientific figure
renderer. Raw arrays, input datasets, native traces, connection details and source-host
paths remain private. Every package file except the manifest itself has its byte
count and SHA-256 in `manifest.json`.

```sh
python -I -B replay.py --output /tmp/deepseek41-aggregate-check.json
python -I -B plot.py --output /tmp/deepseek41-figure
```

Run with a writable output destination outside this immutable archive. The replay
uses only the Python standard library. The renderer requires NumPy and Matplotlib.
The replay verifies package hashes, paired coverage, GPU performance statistics and
all nine NPU performance cells: 45 measured requests, 18 warmups, 63 requests,
959 nonempty forwards and 63 separate untimed cleanup ticks per rank. It recomputes
medians, extrema, arithmetic means and sample standard deviations from the retained
five measurements in each cell, including driver wall statistics.

These commands check aggregate consistency and reproduce the core figure. They do
not recreate model execution or replace the independent private raw numerical
comparison replay using eleven frozen tools. That private replay retains absolute
artifact dependencies and is not fully relocatable.

The numerical inventory retains 191/275 paired cells, 84 GPU gaps and six NPU gaps.
Explicit training, standalone component, framework and padding gaps remain in
`gap-ledger.json`. Framework constructor rejection occurs before accelerator work
and provides compatibility evidence, not training numerical coverage. Runtime
facts distinguish main GPU inference from the separate framework-entrypoint probes.

NPU timings describe the measured adapted BF16 TP8/EP8 offload48 configuration.
GPU timings describe its original mixed-format configuration. These measurements
provide no clean hardware latency ratio or numerical acceptance. NPU fixed32 cached
decode uses max-rank of each rank's sum of 32 cached execute_model spans, excludes
prefill, and differs from a sum of stepwise rank maxima. The original head and CPU
offload are included. Timing and allocator memory boundaries are defined in
`npu-performance.json`; phase peaks and request peaks differ for decode. Cleanup
summaries describe recorded observations for the exact completed native lifetime,
not later device state or whole-host exclusivity. No numerical acceptance threshold
was supplied; completed evidence delivery does not imply all numerical chains pass.
