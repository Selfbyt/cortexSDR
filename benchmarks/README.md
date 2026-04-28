# CortexSDR Benchmarks

Native benchmark runner for compression, extraction, and inference metrics.

## Build

```powershell
cmake -S . -B build
cmake --build build --config Debug --target cortexsdr_bench
```

## Configure

Copy and edit the suite template:

```powershell
Copy-Item benchmarks\suite.example.json benchmarks\suite.local.json
```

Set your model paths and formats in `benchmarks/suite.local.json`.

## Run

```powershell
.\build\Debug\cortexsdr_bench.exe --config benchmarks\suite.local.json
```

Optional output path:

```powershell
.\build\Debug\cortexsdr_bench.exe --config benchmarks\suite.local.json --output benchmarks\results\my-run.json
```

## Captured Metrics

- Compression:
  - wall time (`compress.elapsed_ms_wall`)
  - parser-reported size and ratio stats from `cortexsdr_ai_compression_cli`
  - measured file size ratio (`compress.compression_ratio_file_measured`)
  - size reduction percent (`compress.size_reduction_percent`)
  - input throughput (`compress.input_throughput_mb_per_sec`)
- Extraction:
  - wall time (`extract.elapsed_ms_wall`)
  - extracted directory size (`extract.extracted_size_bytes`)
  - extraction throughput (`extract.throughput_mb_per_sec`)
  - expansion ratio vs `.sdr` (`extract.expansion_ratio_vs_sdr`)
- Inference:
  - wall time (`inference.elapsed_ms_wall`)
  - profile JSON from `cortexsdr_text_cli --profile`
  - profile TPM (`inference.tokens_per_min`, derived from profile `tokens_per_sec`)
  - wall TPM (`inference.generated_tokens_per_min_wall`)
  - wrapper overhead (`inference.wrapper_overhead_ms`)

## Summary Block

Each run includes a `summary` object with:

- status counts (`ok`/`error`)
- latency distributions (`mean`, `min`, `max`, `p50`, `p95`) for compress/extract/inference
- measured compression ratio distribution
- inference wall TPM distribution

## Suite Options

- Global:
  - `repeats`: number of recorded runs per test
  - `warmup_runs`: warmup runs before recorded runs
- Per test (optional override):
  - `repeats`
  - `warmup_runs`

## Notes

- Keep repeats >= 3 for stable comparisons.
- Run with fixed hardware load and identical model files.
- For LLM comparisons, keep prompt and max length constant.
