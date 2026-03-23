# MMseqs2-GPU Integration for Boltz

This document describes how to use local GPU-accelerated MMseqs2 for MSA generation in Boltz, replacing the ColabFold server API.

## Overview

By default, Boltz uses the ColabFold server (`api.colabfold.com`) for MSA generation. This works well but has limitations:
- Rate limits on the public server
- Requires internet access
- Cannot batch large numbers of inputs efficiently
- Latency from network round-trips

The MMseqs2-GPU integration provides a local alternative that:
- Runs MSA search on your own GPU, no internet required
- Batches all unique sequences in a single GPU call
- Pipelines GPU search with CPU post-processing for maximum throughput
- Supports multi-GPU for large-scale screening

This approach is adapted from [AlphaFast](https://github.com/RomeroLab/alphafast), which integrates MMseqs2-GPU into AlphaFold 3.

---

## Quick Start

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install Boltz with CUDA
uv pip install -e ".[cuda]"

# 3. Install MMseqs2 with GPU support
wget https://mmseqs.com/latest/mmseqs-linux-gpu.tar.gz
tar xzf mmseqs-linux-gpu.tar.gz
sudo cp mmseqs/bin/mmseqs /usr/local/bin/

# 4. Set up databases (~150 GB for ColabFold mode)
./scripts/setup_boltz_mmseqs_dbs.sh /data/boltz_dbs

# 5. Run predictions with local GPU MSA
boltz predict input.yaml --use_mmseqs_gpu --mmseqs_db_dir /data/boltz_dbs
```

---

## Installation Guide

### A. Installing Original Boltz (for a fresh VM)

This sets up the standard Boltz with ColabFold server-based MSA generation.

```bash
# 1. System prerequisites
sudo apt-get update && sudo apt-get install -y \
    git python3 python3-dev wget

# 2. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env  # or restart shell

# 3. Create and activate virtual environment
uv venv ~/boltz-env
source ~/boltz-env/bin/activate

# 4. Install PyTorch with CUDA (adjust for your CUDA version)
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# 5. Clone and install Boltz
git clone https://github.com/jwohlwend/boltz.git
cd boltz
uv pip install -e ".[cuda]"

# 6. Test with a simple protein prediction (uses ColabFold server)
boltz predict examples/prot.yaml --use_msa_server --out_dir ./test_output

# 7. Check output
ls test_output/boltz_results_prot/predictions/
```

### B. Installing Boltz with MMseqs2-GPU (Lightning-Boltz)

This sets up Boltz with local GPU-accelerated MSA generation.

```bash
# 1. System prerequisites
sudo apt-get update && sudo apt-get install -y \
    git python3 python3-dev wget zstd

# 2. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env  # or restart shell

# 3. Create and activate virtual environment
uv venv ~/boltz-env
source ~/boltz-env/bin/activate

# 4. Install PyTorch with CUDA
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# 5. Clone and install Boltz (this repository with MMseqs2-GPU support)
git clone <this-repo-url>
cd lb-ben-dev
uv pip install -e ".[cuda]"

# 6. Install MMseqs2 with GPU support
wget https://mmseqs.com/latest/mmseqs-linux-gpu.tar.gz
tar xzf mmseqs-linux-gpu.tar.gz
sudo cp mmseqs/bin/mmseqs /usr/local/bin/
mmseqs version  # verify installation

# 7. Set up databases (see Database Setup section below)
./scripts/setup_boltz_mmseqs_dbs.sh /data/boltz_dbs

# 8. Test with local GPU MSA
boltz predict examples/prot.yaml \
    --use_mmseqs_gpu \
    --mmseqs_db_dir /data/boltz_dbs \
    --out_dir ./test_output

# 9. Check output
ls test_output/boltz_results_prot/predictions/
```

---

## Database Setup

### Option 1: ColabFold Databases (Recommended)

The ColabFold databases are smaller and match what the ColabFold server uses.

```bash
# Basic setup (UniRef30 + ColabFold env DB, ~150 GB)
./scripts/setup_boltz_mmseqs_dbs.sh /data/boltz_dbs

# From HuggingFace (pre-indexed tarballs, no conversion needed — faster)
./scripts/setup_boltz_mmseqs_dbs.sh /data/boltz_dbs --source huggingface

# With UniProt for multi-chain pairing (~250 GB)
./scripts/setup_boltz_mmseqs_dbs.sh /data/boltz_dbs --with-uniprot
```

**Databases downloaded:**
| Database | Size (compressed) | Purpose |
|----------|------------------|---------|
| UniRef30 2302 | ~24 GB | Primary protein homolog search |
| ColabFold env DB | ~110 GB | Environmental sequences (BFD + MGnify + MetaEuk) |
| UniProt (optional) | ~65 GB | Taxonomy-based pairing for multi-chain complexes |

### Option 2: AlphaFold3 Databases

Use if you already have AlphaFast/AlphaFold3 databases, or want the most comprehensive coverage.

```bash
# Fresh download (~800 GB)
./scripts/setup_boltz_mmseqs_dbs.sh /data/boltz_dbs --mode alphafold3

# If you already have AlphaFast databases set up, point directly to them
boltz predict input.yaml \
    --use_mmseqs_gpu \
    --mmseqs_db_dir /path/to/alphafast/mmseqs/
```

**Databases:**
| Database | Size (raw FASTA) | Purpose |
|----------|-----------------|---------|
| UniRef90 | 67 GB | Primary protein homolog search |
| MGnify | 120 GB | Metagenomic sequences |
| Small BFD | 17 GB | Big Fantastic Database subset |
| UniProt | 101 GB | Taxonomy-based pairing |

### Reusing AlphaFast Databases

If you already have AlphaFast set up with padded databases, you can reuse them directly:

```bash
# Point to your existing AlphaFast mmseqs directory
boltz predict input.yaml \
    --use_mmseqs_gpu \
    --mmseqs_db_dir /path/to/alphafast_databases/mmseqs/
```

The code auto-detects databases by looking for `*_padded.dbtype` files.

---

## Usage

### Basic Usage

```bash
# Single protein prediction
boltz predict examples/prot.yaml \
    --use_mmseqs_gpu \
    --mmseqs_db_dir /data/boltz_dbs

# Multi-chain complex (pairing requires UniProt database)
boltz predict examples/multimer.yaml \
    --use_mmseqs_gpu \
    --mmseqs_db_dir /data/boltz_dbs

# Protein-ligand prediction
boltz predict examples/ligand.yaml \
    --use_mmseqs_gpu \
    --mmseqs_db_dir /data/boltz_dbs
```

### Environment Variable

Set `BOLTZ_MMSEQS_DB_DIR` to avoid passing `--mmseqs_db_dir` every time:

```bash
export BOLTZ_MMSEQS_DB_DIR=/data/boltz_dbs
boltz predict input.yaml --use_mmseqs_gpu
```

### Advanced Options

```bash
boltz predict input.yaml \
    --use_mmseqs_gpu \
    --mmseqs_db_dir /data/boltz_dbs \
    --mmseqs_binary /usr/local/bin/mmseqs \    # Custom mmseqs path
    --mmseqs_gpu_device 0 \                    # Use specific GPU
    --mmseqs_threads 32 \                      # CPU threads
    --mmseqs_sensitivity 7.5 \                 # Search sensitivity (1-7.5)
    --mmseqs_temp_dir /scratch \               # Fast local storage (HPC)
    --preprocessing-threads 8                  # Parallel input processing
```

### Multi-GPU Mode

For multi-GPU systems, use separate processes with GPU device assignment:

```bash
# GPU 0: MSA search for first batch
CUDA_VISIBLE_DEVICES=0 boltz predict batch1/ \
    --use_mmseqs_gpu \
    --mmseqs_db_dir /data/boltz_dbs \
    --mmseqs_gpu_device 0

# GPU 1: MSA search for second batch (in parallel)
CUDA_VISIBLE_DEVICES=1 boltz predict batch2/ \
    --use_mmseqs_gpu \
    --mmseqs_db_dir /data/boltz_dbs \
    --mmseqs_gpu_device 1
```

### CLI Reference

| Flag | Description | Default |
|------|-------------|---------|
| `--use_mmseqs_gpu` | Enable local MMseqs2-GPU MSA generation | `False` |
| `--mmseqs_db_dir` | Directory with padded databases | `$BOLTZ_MMSEQS_DB_DIR` |
| `--mmseqs_binary` | Path to mmseqs executable | Auto-detected |
| `--mmseqs_gpu_device` | GPU device index | All GPUs |
| `--mmseqs_threads` | CPU threads for non-GPU ops | All cores |
| `--mmseqs_sensitivity` | Search sensitivity (1-7.5) | `7.5` |
| `--mmseqs_temp_dir` | Fast temp storage directory | System temp |

---

## Testing the Implementation

### 1. Verify MMseqs2 Installation

```bash
# Check binary is accessible
mmseqs version

# Verify GPU support (should show CUDA info)
mmseqs search -h 2>&1 | grep -i gpu
```

### 2. Verify Database Setup

```bash
# List available padded databases
ls -la /data/boltz_dbs/*_padded.dbtype

# Test a simple search manually
echo ">test
MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH" > /tmp/test.fasta
mmseqs createdb /tmp/test.fasta /tmp/testdb
mmseqs search /tmp/testdb /data/boltz_dbs/uniref30_2302_padded /tmp/result /tmp/tmp -a --gpu 1
echo "Search completed successfully!"
```

### 3. Compare Server vs Local MSA

Run the same prediction with both methods and compare:

```bash
# Method 1: ColabFold server (baseline)
boltz predict examples/prot.yaml \
    --use_msa_server \
    --out_dir ./test_server

# Method 2: Local MMseqs2-GPU
boltz predict examples/prot.yaml \
    --use_mmseqs_gpu \
    --mmseqs_db_dir /data/boltz_dbs \
    --out_dir ./test_local

# Compare outputs
diff -r test_server/boltz_results_prot/predictions/ \
        test_local/boltz_results_prot/predictions/
```

### 4. Batch Processing Test

```bash
# Create a directory with multiple input files
mkdir -p test_batch/
cp examples/prot.yaml test_batch/prot1.yaml
cp examples/prot.yaml test_batch/prot2.yaml

# Run batch prediction
time boltz predict test_batch/ \
    --use_mmseqs_gpu \
    --mmseqs_db_dir /data/boltz_dbs \
    --out_dir ./test_batch_output \
    --preprocessing-threads 4
```

### 5. Benchmarking Speed

```bash
# Time the MSA generation only (add --accelerator cpu to skip GPU inference)
time boltz predict examples/prot.yaml \
    --use_mmseqs_gpu \
    --mmseqs_db_dir /data/boltz_dbs \
    --out_dir ./bench_local \
    --accelerator cpu

time boltz predict examples/prot.yaml \
    --use_msa_server \
    --out_dir ./bench_server \
    --accelerator cpu
```

---

## Architecture

### Data Flow

```
Input YAML/FASTA
    │
    ▼
parse_yaml() / parse_fasta()
    │
    ▼
process_input()
    │
    ├─ use_msa_server: ColabFold API → run_mmseqs2()
    │                                    │
    └─ use_mmseqs_gpu: Local GPU    → compute_msa_local()
                                         │
                                         ├─ pipelined_search()
                                         │   ├─ GPU: search DB1 (sync)
                                         │   ├─ CPU: postprocess DB1 (async)
                                         │   ├─ GPU: search DB2 (sync)
                                         │   ├─ CPU: postprocess DB2 (async)
                                         │   └─ ...
                                         │
                                         ├─ batch_search() [UniProt, if multi-chain]
                                         │   └─ taxonomy-based pairing
                                         │
                                         └─ Write CSV (key, sequence)
    │
    ▼
Tokenizer → Featurizer → Model → Predictions
```

### Key Files

| File | Description |
|------|-------------|
| `src/boltz/data/msa/mmseqs_local.py` | Core MMseqs2-GPU integration |
| `src/boltz/data/msa/mmseqs2.py` | Original ColabFold server client |
| `src/boltz/main.py` | CLI and pipeline orchestration |
| `scripts/setup_boltz_mmseqs_dbs.sh` | Database setup script |

### Batching Strategy

When processing multiple inputs, sequences are batched for efficient GPU utilization:

1. All unique protein sequences across inputs are collected
2. A single shared query database is created (`mmseqs createdb`)
3. GPU searches run sequentially per target database to avoid OOM
4. CPU post-processing (result2msa + unpackdb) runs in parallel with subsequent GPU searches
5. Results are mapped back to individual sequences via the `.lookup` file

### Paired MSA (Multi-Chain)

For multi-chain complexes, paired MSA is generated via taxonomy-based alignment:

1. Each chain's sequence is searched against UniProt
2. Taxonomy IDs are extracted from hit headers (OX= field)
3. Common taxonomies across all chains are identified
4. Hits are aligned by taxonomy: position i across all chains corresponds to the same organism
5. The position index serves as the pairing key in the output CSV

---

## Troubleshooting

### MMseqs2 binary not found

```
MMseqs2 binary not found. Install with:
  wget https://mmseqs.com/latest/mmseqs-linux-gpu.tar.gz
  tar xzf mmseqs-linux-gpu.tar.gz
  sudo cp mmseqs/bin/mmseqs /usr/local/bin/
```

### No databases found

```
No MMseqs2 databases found in /data/boltz_dbs.
```

Run the setup script: `./scripts/setup_boltz_mmseqs_dbs.sh /data/boltz_dbs`

### GPU out of memory during MSA search

Try reducing the search scope or using `--mmseqs_temp_dir` for fast local storage:

```bash
boltz predict input.yaml \
    --use_mmseqs_gpu \
    --mmseqs_db_dir /data/boltz_dbs \
    --mmseqs_temp_dir /scratch \
    --mmseqs_sensitivity 5.5
```

### HPC clusters with slow network storage

Use `--mmseqs_temp_dir` to point to fast local storage (e.g., `/scratch`, `/tmp`):

```bash
boltz predict input.yaml \
    --use_mmseqs_gpu \
    --mmseqs_db_dir /data/boltz_dbs \
    --mmseqs_temp_dir /scratch/$USER/mmseqs_tmp
```

This can provide 10-13x speedup on clusters with slow network-attached storage.

### Cannot use both --use_msa_server and --use_mmseqs_gpu

These flags are mutually exclusive. Choose one MSA generation method:
- `--use_msa_server`: Uses the ColabFold server (requires internet)
- `--use_mmseqs_gpu`: Uses local GPU-accelerated search (requires local databases)

---

## Deployment Options

Lightning-Boltz supports multiple deployment environments:

| Environment | Guide | Best for |
|---|---|---|
| **Bare metal** | This document + [install script](../scripts/install_lightning_boltz.sh) | Dedicated servers, workstations |
| **HPC (SLURM)** | [docs/HPC.md](HPC.md) | University clusters, shared compute |
| **Multi-GPU** | [scripts/run_multigpu.sh](../scripts/run_multigpu.sh) | High-throughput screening |
| **Serverless** | [modal/README.md](../modal/README.md) | On-demand GPU, no hardware |
| **Container** | [Dockerfile](../Dockerfile) | Reproducible environments |

### Quick links

- **Install script**: `./scripts/install_lightning_boltz.sh` — one-command setup on Linux
- **Environment template**: `scripts/boltz.env.template` — configure database paths
- **Database setup**: `./scripts/setup_boltz_mmseqs_dbs.sh /data/boltz_dbs` — download databases
- **SLURM templates**: `scripts/slurm/` — ready-to-submit job scripts
