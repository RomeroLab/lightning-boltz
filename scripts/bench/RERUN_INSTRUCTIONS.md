# Benchmark Rerun Instructions

## Current Failure Mode

The `batch_size=1` L40S run completed MSA/preprocessing and then failed during
mmCIF writing for a protein-ligand target:

```text
ValueError: Assemblies reference asym IDs that don't have coordinates...
asym IDs C, D
```

This is an output writer issue, not an MMseqs issue. The benchmark harness now
defaults to `--output_format pdb` to avoid this mmCIF validation failure.

## Missing CCD Components

The run skipped `175` inputs during preprocessing. Most missing CCD IDs are
newer 5-character PDB ligand components such as `A1C65`, `A1JUW`, and `A1C3I`.
Boltz's bundled molecule cache contains standard components such as `D8U.pkl`
and `ATP.pkl`, but not these newer 5-character components.

The processed benchmark set for that run is therefore:

```text
1873 = 2048 total inputs - 175 skipped ligand inputs
```

## Rerun The Failed Batch

Use `--force` to rerun full inference with PDB output while reusing the existing
processed/MSA files. The harness preserves the old failed log as
`*.failed_TIMESTAMP.log` and records its MSA time as `previous_msa_time_s` in the
new result JSON.

```bash
bash scripts/bench/run_lightning_local_docker.sh \
  --input-dir bench_runs/inputs/n2048_flat \
  --db-dir /opt/lightning-boltz-data \
  --image lightning-boltz:bench \
  --replicates 1 \
  --force
```

This command runs all batch sizes in one go:

```text
1 8 16 32 64 128 256 512 1024 2048
```

To rerun only `batch_size=1`:

```bash
bash scripts/bench/run_lightning_local_docker.sh \
  --input-dir bench_runs/inputs/n2048_flat \
  --db-dir /opt/lightning-boltz-data \
  --image lightning-boltz:bench \
  --replicates 1 \
  --batch-sizes "1" \
  --force
```

## Triplicate Runs On Three Local GPUs

Run one technical replicate per GPU in separate `tmux` sessions with separate
output directories.

```bash
tmux new -s boltz_rep1 'CUDA_VISIBLE_DEVICES=0 bash scripts/bench/run_lightning_local_docker.sh --input-dir bench_runs/inputs/n2048_flat --db-dir /opt/lightning-boltz-data --image lightning-boltz:bench --gpu-device 0 --replicates 1 --out-dir bench_runs/lightning_l40s_rep1'
```

```bash
tmux new -s boltz_rep2 'CUDA_VISIBLE_DEVICES=1 bash scripts/bench/run_lightning_local_docker.sh --input-dir bench_runs/inputs/n2048_flat --db-dir /opt/lightning-boltz-data --image lightning-boltz:bench --gpu-device 1 --replicates 1 --out-dir bench_runs/lightning_l40s_rep2'
```

```bash
tmux new -s boltz_rep3 'CUDA_VISIBLE_DEVICES=2 bash scripts/bench/run_lightning_local_docker.sh --input-dir bench_runs/inputs/n2048_flat --db-dir /opt/lightning-boltz-data --image lightning-boltz:bench --gpu-device 2 --replicates 1 --out-dir bench_runs/lightning_l40s_rep3'
```

## Summarize Results

```bash
python3 scripts/bench/summarize_benchmarks.py \
  --root bench_runs \
  --out bench_results/summary.csv
```
