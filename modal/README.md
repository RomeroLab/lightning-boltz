# Boltz-2 on Modal (Serverless GPU)

Run Boltz-2 protein structure prediction on [Modal](https://modal.com) cloud GPUs with MMseqs2-GPU for local MSA generation. No GPU hardware to manage — pay per second of compute.

## Quick Start

### 1. Install Modal

```bash
pip install modal
modal setup  # authenticate
```

### 2. Set up databases (one-time)

**Option A: Download pre-built from HuggingFace (recommended)**

Downloads pre-built MMseqs2-GPU databases directly to your Modal volume. This is the fastest option — no compilation or indexing needed.

```bash
# Default repo
modal run modal/upload_dbs.py --from-hf

# Or specify a custom HuggingFace dataset repo
modal run modal/upload_dbs.py --from-hf --hf-repo RomeroLab-Duke/lightning-boltz-data
```

**Option B: Build from source**

Downloads raw databases and builds MMseqs2 GPU-padded indexes from scratch. Takes several hours but doesn't require a pre-built HF repo.

```bash
# ColabFold databases (~150 GB, recommended)
modal run modal/upload_dbs.py

# AlphaFold3 databases (~800 GB)
modal run modal/upload_dbs.py --mode alphafold3

# ColabFold + UniProt for paired MSA
modal run modal/upload_dbs.py --with-uniprot
```

This only needs to run once. The databases persist on Modal's volume storage.

### 3. Set up model checkpoints (one-time)

**Option A: Download inside Modal**

```bash
modal run modal/benchmark.py --download-models
```

**Option B: Upload from local cache**

```bash
# Check what's already on the volume
modal run modal/upload_models.py --check

# Upload from local ~/.boltz
modal run modal/upload_models.py --local-dir ~/.boltz

# Dry run first
modal run modal/upload_models.py --local-dir ~/.boltz --dry-run
```

### 4. Run benchmark

```bash
# Run with default settings (A100-80GB)
modal run modal/benchmark.py --run

# Custom batch size
modal run modal/benchmark.py --run --batch-size 512

# Custom input directory
modal run modal/benchmark.py --run --input-dir examples/set_512_boltz
```

## Status Check

```bash
# Check databases + checkpoints
modal run modal/benchmark.py --status

# Check databases only
modal run modal/upload_dbs.py --check

# Check checkpoints only
modal run modal/upload_models.py --check
```

## Scripts

| Script | Purpose |
|--------|---------|
| `upload_dbs.py` | Download MMseqs2 databases (HuggingFace or from source) |
| `upload_models.py` | Upload local model checkpoints to Modal volume |
| `benchmark.py` | Run Boltz-2 benchmark on Modal A100 GPUs |

## How It Works

1. Input files are read locally and sent to Modal
2. Modal spins up a GPU container with Boltz + MMseqs2-GPU pre-installed
3. Databases are mounted from Modal persistent volumes (no re-download)
4. Prediction runs on the remote GPU
5. Results are returned to the caller

## Architecture

```
Modal Volumes
├── boltz-mmseqs-dbs    # MMseqs2 GPU-padded databases
│   ├── uniref30_*      #   ColabFold mode
│   ├── colabfold_*     #   ColabFold mode
│   └── *_padded*       #   AlphaFold3 mode
└── boltz-cache         # Model checkpoints + CCD data
    ├── boltz2_conf.ckpt
    ├── boltz2_aff.ckpt
    └── mols/
```

## Cost Estimate

| GPU | ~Cost/hour | Typical prediction time |
|-----|-----------|------------------------|
| A10G | $1.10 | 8-15 min |
| A100 | $3.00 | 3-8 min |

Plus ~$0.15/GB/month for database storage (~150 GB ColabFold = ~$22/month).

## Troubleshooting

### "No databases found"

Run `modal run modal/upload_dbs.py --from-hf` first.

### Timeout errors

Increase `timeout` in `benchmark.py` for large protein sets (default: 4 hours).

### Want to use different databases?

Use `--mode alphafold3` with `upload_dbs.py` to download AlphaFold3 databases instead of ColabFold.

## Benchmark Output

The benchmark parses timing information from Boltz's `[Benchmark]` output:

```
[Benchmark] MSA generation + preprocessing: 42.3s
[Benchmark] Inference: 120.5s
[Benchmark] Total: 162.8s
[Benchmark] Num inputs: 10
[Benchmark] Avg per input: 16.28s
```
