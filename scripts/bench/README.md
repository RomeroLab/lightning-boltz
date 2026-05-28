# Boltz-2 Benchmark Harness

This directory contains runners for benchmarking accelerated Lightning-Boltz and official Boltz-2 on `bench_data/n2048`.

## Prepare Inputs

`boltz predict` expects a flat input directory. Prepare one from the nested benchmark data:

```bash
python3 scripts/bench/prepare_inputs.py \
  --source bench_data/n2048 \
  --out bench_runs/inputs/n2048_flat \
  --clean
```

For a smoke test, add `--limit 8 --expected-count 0`.

## Local L40S Docker

Use `--use_colabfold_search` in scripts for this repo. It is the current CLI
name for the local ColabFold-style MSA pipeline, which uses MMseqs2-GPU for the
protein search. `--use_mmseqs_gpu` is kept as an alias in `src/boltz/main.py`,
but the benchmark scripts use the canonical name to avoid confusion.

Build the accelerated image if needed:

```bash
docker build -t lightning-boltz .
```

Run the accelerated single-GPU batch-size sweep. Benchmark runners default to
`--output_format pdb`, skip successful result JSONs, and resume failed result
JSONs without `--override` so existing predictions are reused.

```bash
bash scripts/bench/run_lightning_local_docker.sh \
  --input-dir bench_runs/inputs/n2048_flat \
  --db-dir /path/to/boltz_dbs \
  --replicates 1
```

Use `--force` to overwrite an existing successful result. Use
`--no-resume-failed` to leave failed result JSONs untouched.

### Size-Matched Clean Subsets

The original full sweep runs all batch sizes against all 2048 inputs, which is
too expensive for local iteration. Generate clean subsets where the input count
matches the MMseqs batch size:

```bash
python3 scripts/bench/make_scaled_subsets.py \
  --flat-input-dir bench_runs/inputs/n2048_flat \
  --failed-log bench_runs/lightning_l40s/logs/rep_1_batch_1.log \
  --out-root bench_runs/inputs/scaled_valid \
  --problematic-out bench_runs/inputs/problematic_ccd_smoke16 \
  --clean
```

This creates `valid_n1`, `valid_n4`, `valid_n8`, ..., `valid_n2048` when enough
valid inputs are available. The `n1` set preferentially uses a protein-ligand
input. The `n4` set excludes protein-RNA. Larger sets are sampled
proportionally across categories with tie preference:
`protein_ligand > protein_protein > monomer > protein_dna > protein_rna`.

If `valid_n2048` is missing because the original 2048 inputs include ligands
absent from the Boltz molecule cache, oversample from RCSB and reject missing
CCD entries before writing the exact 2048-input subset:

```bash
python3 scripts/bench/build_valid_n2048_from_rcsb.py \
  --mols-dir "${BOLTZ_CACHE:-$HOME/.boltz}/mols" \
  --out-dir bench_runs/inputs/scaled_valid/valid_n2048 \
  --metadata-out bench_runs/inputs/scaled_valid/valid_n2048_metadata.json \
  --clean
```

Run the size-matched local sweep:

```bash
bash scripts/bench/run_lightning_local_scaled_docker.sh \
  --input-root bench_runs/inputs/scaled_valid \
  --db-dir /opt/lightning-boltz-data \
  --image lightning-boltz:bench \
  --gpu-device 7 \
  --replicate 1
```

The scaled runner now checks that each `valid_n<size>` directory contains
exactly `<size>` YAML files before starting the container.

For a quick smoke run:

```bash
bash scripts/bench/run_lightning_local_scaled_docker.sh \
  --input-root bench_runs/inputs/scaled_valid \
  --db-dir /opt/lightning-boltz-data \
  --image lightning-boltz:bench \
  --gpu-device 7 \
  --replicate 1 \
  --sizes "1 4 8"
```

Run official Boltz on the problematic CCD smoke set:

```bash
bash scripts/bench/run_boltz_official_local_docker.sh \
  --input-dir bench_runs/inputs/problematic_ccd_smoke16 \
  --image boltz-official-bench \
  --gpu-device 7 \
  --replicates 1 \
  --extra-args "--sampling_steps 1 --recycling_steps 1 --diffusion_samples 1"
```

Run the official Boltz-2 single-GPU baseline with the MSA server. This records `batch_size=1` because official Boltz does not use local MMseqs batching:

```bash
bash scripts/bench/run_boltz_official_local_docker.sh \
  --input-dir bench_runs/inputs/n2048_flat \
  --replicates 1
```

## H200 Singularity/SLURM

The H200 runners use `#SBATCH --partition=scavenger-h200` and `#SBATCH --account=scavenger-h200`.

Build or pull the accelerated Singularity image:

```bash
singularity build lightning-boltz.sif singularity/lightning-boltz.def
# or, from the published Docker benchmark image:
singularity build lightning-boltz_bench-cf37c3e.sif docker://romerolabduke/lightning-boltz:bench-cf37c3e
```

Build the official Boltz image from `https://github.com/jwohlwend/boltz.git`:

```bash
singularity build boltz-official.sif singularity/boltz-official.def
```

Run the accelerated single-H200 batch-size sweep:

```bash
sbatch --export=ALL,INPUT_DIR=/path/to/n2048_flat,DB_DIR=/path/to/boltz_dbs,SIF=/path/to/lightning-boltz.sif,REPLICATES=1 \
  scripts/bench/run_lightning_h200_single.sbatch
```

Run the size-matched single-H200 sweep:

```bash
sbatch --export=ALL,INPUT_ROOT=/path/to/scaled_valid,DB_DIR=/path/to/boltz_dbs,SIF=/path/to/lightning-boltz.sif,REPLICATE=1 \
  scripts/bench/run_lightning_h200_scaled_single.sbatch
```

Preferred staged H200 submission, matching the lab SLURM directive style and
copying YAML inputs plus the Boltz cache to node-local storage:

```bash
sbatch \
  --export=ALL,INPUT_ROOT=/work/bcp28/lightning-boltz/bench_runs/inputs/scaled_valid,DB_DIR=/work/bcp28/boltz_dbs,SIF=/work/bcp28/images/lightning-boltz.sif \
  scripts/bench/run_h200_scaled_single_staged.sbatch
```

Run accelerated multi-H200 jobs at batch size `256`:

```bash
sbatch --gres=gpu:h200:2 --export=ALL,INPUT_DIR=/path/to/n2048_flat,DB_DIR=/path/to/boltz_dbs,SIF=/path/to/lightning-boltz.sif,NUM_GPUS=2,REPLICATES=1 \
  scripts/bench/run_lightning_h200_multigpu.sbatch

sbatch --gres=gpu:h200:4 --export=ALL,INPUT_DIR=/path/to/n2048_flat,DB_DIR=/path/to/boltz_dbs,SIF=/path/to/lightning-boltz.sif,NUM_GPUS=4,REPLICATES=1 \
  scripts/bench/run_lightning_h200_multigpu.sbatch

sbatch --gres=gpu:h200:8 --export=ALL,INPUT_DIR=/path/to/n2048_flat,DB_DIR=/path/to/boltz_dbs,SIF=/path/to/lightning-boltz.sif,NUM_GPUS=8,REPLICATES=1 \
  scripts/bench/run_lightning_h200_multigpu.sbatch
```

Preferred staged multi-H200 submission. Set `BATCH_SIZE` to the fastest
single-H200 batch size and use the matching `valid_n<BATCH_SIZE>` input set:

```bash
for g in 2 4 6 8; do
  sbatch --gres=gpu:h200:${g} \
    --export=ALL,INPUT_DIR=/work/bcp28/lightning-boltz/bench_runs/inputs/scaled_valid/valid_n512,DB_DIR=/work/bcp28/boltz_dbs,SIF=/work/bcp28/images/lightning-boltz.sif,NUM_GPUS=${g},BATCH_SIZE=512 \
    scripts/bench/run_h200_multigpu_staged.sbatch
done
```

Run the official Boltz-2 single-H200 baseline:

```bash
sbatch --export=ALL,INPUT_DIR=/path/to/n2048_flat,SIF=/path/to/boltz-official.sif,REPLICATES=1 \
  scripts/bench/run_boltz_official_h200_single.sbatch
```

Use `valid_n1` or `valid_n8` for the low-rate official baseline. Prior local
L40S testing showed `valid_n16` can already hit ColabFold MSA server rate
limits. Use `valid_n512` as the deliberate stress test to demonstrate API rate
limiting:

```bash
sbatch \
  --export=ALL,INPUT_DIR=/work/bcp28/lightning-boltz/bench_runs/inputs/scaled_valid/valid_n8,SIF=/work/bcp28/images/boltz-official.sif,OUT_DIR=bench_runs/boltz_official_h200_n8 \
  scripts/bench/run_boltz_official_h200_staged.sbatch

sbatch \
  --export=ALL,INPUT_DIR=/work/bcp28/lightning-boltz/bench_runs/inputs/scaled_valid/valid_n512,SIF=/work/bcp28/images/boltz-official.sif,OUT_DIR=bench_runs/boltz_official_h200_n512 \
  scripts/bench/run_boltz_official_h200_staged.sbatch
```

## Summarize

```bash
python3 scripts/bench/summarize_benchmarks.py \
  --root bench_runs \
  --out bench_results/summary.csv
```

Each runner writes raw logs under `logs/` and per-run JSON under `results/`.
