# Lightning-Boltz on HPC (SLURM)

Guide for running Lightning-Boltz on HPC clusters with SLURM job scheduling.

---

## Installation

### Option A: uv install (preferred)

Install Boltz and the MMseqs2-GPU binary into user space. No root access needed.

```bash
# 1. Load modules (adjust for your cluster)
module load cuda/12.1
module load python/3.11

# 2. Install uv (if not already available)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 3. Create virtual environment
uv venv ~/boltz-env
source ~/boltz-env/bin/activate

# 4. Install PyTorch + Boltz
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install -e ".[cuda]"  # from source
# or: uv pip install "boltz[cuda]"  # from PyPI

# 5. Install MMseqs2-GPU binary (no root needed)
wget https://mmseqs.com/latest/mmseqs-linux-gpu.tar.gz
tar xzf mmseqs-linux-gpu.tar.gz
mkdir -p ~/.local/bin
cp mmseqs/bin/mmseqs ~/.local/bin/
rm -rf mmseqs mmseqs-linux-gpu.tar.gz

# 6. Verify
mmseqs version
boltz --help
```

Add to your `~/.bashrc`:
```bash
export PATH="$HOME/.local/bin:$PATH"
```

### Option B: Singularity container (fallback)

If the uv install is problematic (e.g., incompatible system libraries):

```bash
# Option 1: Pull directly from Docker Hub (one-time, ~5 GB)
singularity pull ~/containers/lightning-boltz.sif \
    docker://romerolabduke/lightning-boltz:latest

# Option 2: Build from the definition file (if you want to customize)
singularity build ~/containers/lightning-boltz.sif \
    singularity/lightning-boltz.def

# Run predictions
singularity run --nv ~/containers/lightning-boltz.sif \
    boltz predict input.yaml \
    --use_mmseqs_gpu \
    --mmseqs_db_dir /scratch/$USER/boltz_dbs
```

The `--nv` flag enables GPU passthrough in Singularity. See
[`singularity/lightning-boltz.def`](../singularity/lightning-boltz.def) for the
full definition file.

---

## Database Setup

Databases should be on shared or scratch storage accessible from compute nodes.

```bash
# Interactive session for download (CPU-only, needs internet)
srun --partition=cpu --cpus-per-task=16 --mem=64G --time=12:00:00 --pty bash

# Download ColabFold databases (~150 GB)
./scripts/setup_boltz_mmseqs_dbs.sh /scratch/$USER/boltz_dbs

# Or from HuggingFace (pre-indexed, no conversion needed — faster on clusters
# with good internet connectivity)
./scripts/setup_boltz_mmseqs_dbs.sh /scratch/$USER/boltz_dbs --source huggingface

# Or submit as a batch job
sbatch scripts/slurm/setup_databases.sbatch
```

### Storage recommendations

| Storage type | Use for | Notes |
|---|---|---|
| Shared/project storage | Databases (read-only after setup) | Persistent, accessible from all nodes |
| Scratch | `--mmseqs_temp_dir` | 10-13x speedup over network FS |
| Local NVMe (`/tmp`) | Temp files during prediction | Fastest, but node-local |

**Critical**: Always set `--mmseqs_temp_dir` to fast local storage:
```bash
boltz predict input.yaml \
    --use_mmseqs_gpu \
    --mmseqs_db_dir /scratch/$USER/boltz_dbs \
    --mmseqs_temp_dir /tmp/$USER/mmseqs_tmp
```

### Verifying databases

```bash
./scripts/setup_boltz_mmseqs_dbs.sh /scratch/$USER/boltz_dbs --verify
```

---

## SLURM Job Templates

Ready-to-use job scripts are in `scripts/slurm/`. Copy and modify the `#SBATCH` headers for your cluster.

### Single GPU prediction

```bash
sbatch scripts/slurm/predict_single.sbatch
```

### Multi-GPU prediction

```bash
sbatch scripts/slurm/predict_multigpu.sbatch
```

### Database setup

```bash
sbatch scripts/slurm/setup_databases.sbatch
```

---

## Resource Sizing Guide

| Workload | GPUs | CPUs | RAM | Time (approx) |
|---|---|---|---|---|
| Single protein (< 1000 residues) | 1 | 8 | 32 GB | 5-15 min |
| Single protein (1000-3000 residues) | 1 | 8 | 64 GB | 15-45 min |
| Batch (100 proteins) | 1 | 16 | 64 GB | 2-4 hours |
| Batch (100 proteins) | 4 | 32 | 128 GB | 30-60 min |
| Database setup | 0 | 16 | 64 GB | 4-12 hours |

GPU memory requirements:
- MSA search (MMseqs2-GPU): ~4-8 GB VRAM
- Structure prediction (Boltz): ~16-40 GB VRAM depending on protein size
- Both phases share the same GPU sequentially (PyTorch releases memory after MSA)

---

## CUDA Compatibility

MMseqs2-GPU requires CUDA >= 12.1. Check your cluster's CUDA version:

```bash
nvidia-smi  # shows driver version
nvcc --version  # shows toolkit version (if loaded)
module avail cuda  # list available CUDA modules
```

| NVIDIA Driver | Max CUDA | Compatible? |
|---|---|---|
| >= 530 | CUDA 12.1+ | Yes |
| 525 | CUDA 12.0 | No (need 12.1+) |
| < 525 | CUDA 11.x | No |

---

## Troubleshooting

### "CUDA driver version is insufficient"
Load a newer CUDA module or ask your sysadmin to update the NVIDIA driver.

### Slow MSA search on network filesystem
Set `--mmseqs_temp_dir` to local storage (`/tmp`, `/scratch`, or node-local NVMe).

### Out of memory during structure prediction
- Reduce `--recycling_steps` (default: 3)
- Use `--override_num_samples 1` for fewer samples
- Request a node with more GPU memory

### Module conflicts
If `module load cuda` conflicts with PyTorch's bundled CUDA:
```bash
# Option 1: Don't load CUDA module, PyTorch bundles its own
module purge
module load python/3.11
source ~/boltz-env/bin/activate

# Option 2: Use Singularity container (avoids all module issues)
singularity run --nv ~/containers/lightning-boltz.sif boltz predict ...
```

---

## See Also

- [MMseqs2-GPU Integration Guide](MMSEQS_GPU.md) — detailed usage and architecture
- [scripts/slurm/](../scripts/slurm/) — SLURM job templates
