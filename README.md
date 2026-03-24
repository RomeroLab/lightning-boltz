<p align="center"><img src=".github/logo.png" height="256" /></p>

# Lightning-Boltz

Ultra-high-throughput inference with [Boltz-2](https://doi.org/10.1101/2025.06.14.659707). Replaces the MMSeqs2 web server dependency with local [MMseqs2-GPU](https://github.com/soedinglab/MMseqs2) for GPU-accelerated MSA generation. Inference can run offline with no rate limits or network latency.

Adapted techniques from [AlphaFast](https://github.com/RomeroLab/alphafast), which integrates MMseqs2-GPU into AlphaFold 3.

Check out the AlphaFast [preprint](https://www.biorxiv.org/content/10.64898/2026.02.17.706409v1.article-metrics), the Boltz-2 [preprint](https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1), and the MMSeqs2-GPU [publication](https://www.nature.com/articles/s41592-025-02819-8)!

## Quick Start

### Step 1: Choose Your Compute Environment

| Environment | Requirements | Jump to |
|-------------|-------------|---------|
| Local (Package) |`uv` or `pip`| [Bare Metal Setup](#bare-metal-setup) |
| Local (Docker) | Docker | [Docker Setup](#docker-setup) |
| HPC (Singularity) | Singularity, SLURM | [HPC Setup](#hpc-setup) |
| Modal (Serverless) | Modal Billing Account | [Modal Setup](#modal-setup) |

---

## Bare Metal Setup

### Step 2: Install

```bash
bash scripts/install_lightning_boltz.sh
```

Or step-by-step:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv ~/boltz-env && source ~/boltz-env/bin/activate
uv pip install -e ".[cuda]"

# Install MMseqs2-GPU binary
wget https://mmseqs.com/latest/mmseqs-linux-gpu.tar.gz
tar xzf mmseqs-linux-gpu.tar.gz
mkdir -p ~/.local/bin && cp mmseqs/bin/mmseqs ~/.local/bin/
rm -rf mmseqs mmseqs-linux-gpu.tar.gz
```

### Step 3: Download Databases

> **Important:** Point the path to a fast data drive (NVMe recommended).

```bash
./scripts/setup_boltz_mmseqs_dbs.sh /path/to/databases
```

### Step 4: Create Input

Create a YAML input file. See [`docs/prediction.md`](docs/prediction.md) for the full format reference. Minimal example:

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQ...
```

### Step 5: Run Inference

**Single GPU:**

```bash
boltz predict input.yaml \
    --use_mmseqs_gpu \
    --mmseqs_db_dir /path/to/databases
```

**Multi-GPU:**

```bash
scripts/run_multigpu.sh \
    --input_dir ./inputs \
    --db_dir /path/to/databases \
    --num_gpus 4
```

---

## Docker Setup

### Step 2: Build Container

```bash
docker build -t lightning-boltz .
```

### Step 3: Download Databases

```bash
./scripts/setup_boltz_mmseqs_dbs.sh /path/to/databases
```

### Step 4: Create Input

Create a YAML input file. See [`docs/prediction.md`](docs/prediction.md) for the full format reference. Minimal example:

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQ...
```

### Step 5: Run Inference

```bash
docker run --gpus '"device=0"' \
    --user "$(id -u):$(id -g)" \
    --ipc=host \
    -v /path/to/databases:/dbs \
    -v ./inputs:/inputs \
    -v ./outputs:/outputs \
    lightning-boltz \
    boltz predict /inputs/prot.yaml --use_mmseqs_gpu --mmseqs_db_dir /dbs --out_dir /outputs
```

---

## HPC Setup

### Step 2: Build Container

> **Important:** Most HPC systems use Apptainer/Singularity rather than Docker. Ensure your cache directory is on an appropriately sized volume — home directories are often too small. See [`docs/HPC.md`](docs/HPC.md) for specific guidance.

```bash
# Build from the Dockerfile (requires a machine with Docker access)
docker build -t lightning-boltz .
docker save lightning-boltz -o lightning-boltz.tar

# Convert to Singularity on the HPC node
singularity build lightning-boltz.sif docker-archive://lightning-boltz.tar
```

Or, if you have a container registry, push the Docker image and pull directly:

```bash
singularity pull lightning-boltz.sif docker://your-registry/lightning-boltz:latest
```

### Step 3: Install Databases

> **Important:** Point the path to a high-speed volume. You will need ~**150 GB** free disk space. You may need to edit SLURM directives to match your cluster's configuration.

```bash
# Submit as SLURM job
sbatch scripts/slurm/setup_databases.sbatch /path/to/databases

# Or run directly in an interactive session
./scripts/setup_boltz_mmseqs_dbs.sh /path/to/databases
```

### Step 4: Create Input

Create a YAML input file. See [`docs/prediction.md`](docs/prediction.md) for the full format reference. Minimal example:

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQ...
```

### Step 5: Run

```bash
# Single GPU
sbatch scripts/slurm/predict_single.sbatch

# Multi-GPU
sbatch scripts/slurm/predict_multigpu.sbatch
```

See [`docs/HPC.md`](docs/HPC.md) for installation, resource sizing, and troubleshooting.

---

## Modal Setup

Run Lightning-Boltz on [Modal](https://modal.com) serverless GPUs — no hardware to manage, pay per second of compute.

### Step 2: Install Modal

```bash
pip install modal
modal setup  # authenticate with your Modal account
```

### Step 3: Set Up Databases (one-time)

**Option A: Download pre-built from HuggingFace (recommended)**

Downloads pre-built MMseqs2-GPU databases directly to your Modal volume. No compilation or indexing needed.

```bash
modal run modal/upload_dbs.py --from-hf
```

**Option B: Build from source**

Downloads raw databases and builds MMseqs2 GPU-padded indexes from scratch. Takes several hours.

```bash
# ColabFold databases (~150 GB, recommended)
modal run modal/upload_dbs.py

# AlphaFold3 databases (~800 GB)
modal run modal/upload_dbs.py --mode alphafold3

# ColabFold + UniProt for paired MSA
modal run modal/upload_dbs.py --with-uniprot
```

### Step 4: Set Up Model Checkpoints (one-time)

```bash
# Download inside Modal
modal run modal/benchmark.py --download-models

# Or upload from local cache
modal run modal/upload_models.py --local-dir ~/.boltz
```

### Step 5: Run Predictions

```bash
modal run modal/predict.py --input examples/prot.yaml
```

Or run the full benchmark suite:

```bash
modal run modal/benchmark.py --run
```

### Status Check

```bash
# Check databases + checkpoints
modal run modal/benchmark.py --status

# Check databases only
modal run modal/upload_dbs.py --check
```

### Cost Estimate

| GPU | ~Cost/hour | Typical prediction time |
|-----|-----------|------------------------|
| A10G | $1.10 | 8–15 min |
| A100 | $3.00 | 3–8 min |

Plus ~$0.15/GB/month for database storage (~150 GB ColabFold = ~$22/month).

See [`modal/README.md`](modal/README.md) for architecture details and troubleshooting.

---

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--use_mmseqs_gpu` | `False` | Enable local GPU MSA generation |
| `--mmseqs_db_dir` | `$BOLTZ_MMSEQS_DB_DIR` | Path to MMseqs2 databases |
| `--mmseqs_gpu_device` | All GPUs | GPU device index for MSA search |
| `--mmseqs_threads` | All cores | CPU threads for post-processing |
| `--mmseqs_sensitivity` | `7.5` | Search sensitivity (1-7.5) |
| `--mmseqs_temp_dir` | System temp | Fast temp storage (important on HPC) |
| `--mmseqs_batch_size` | `512` | Max sequences per GPU search batch |
| `--preprocessing-threads` | `1` | Parallel input preprocessing threads |

**Alternative:** `--use_msa_server` uses the ColabFold public server for MSA generation — no databases needed, but rate-limited and requires internet access.

## Citing This Work

If you use this code or the models in your research, please cite:

### AlphaFast
```bibtex
@article{Perry2026.02.17.706409,
 author = {Perry, Benjamin C and Kim, Jeonghyeon and Romero, Philip A},
 title = {AlphaFast: High-throughput AlphaFold 3 via GPU-accelerated MSA construction},
 year = {2026},
 doi = {10.64898/2026.02.17.706409},
 publisher = {Cold Spring Harbor Laboratory},
 abstract = {AlphaFold 3 (AF3) enables accurate biomolecular modeling but is limited by slow, CPU-bound multiple sequence alignment (MSA) generation. We introduce AlphaFast, a drop-in framework that integrates GPU-accelerated MMseqs2 sequence search to remove this bottleneck. AlphaFast achieves a 68.5x speedup in MSA construction and a 22.8x reduction in end-to-end runtime on a single GPU, and delivers predictions in 8 seconds per input on four GPUs while maintaining indistinguishable structural accuracy. A serverless deployment enables structure prediction for as little as $0.035 per input. Code is available at https://github.com/RomeroLab/alphafast.},
 URL = {https://www.biorxiv.org/content/early/2026/02/18/2026.02.17.706409},
 journal = {bioRxiv}
}
```

### Boltz Papers
```bibtex
@article{passaro2025boltz2,
  author = {Passaro, Saro and Corso, Gabriele and Wohlwend, Jeremy and Reveiz, Mateo and Thaler, Stephan and Somnath, Vignesh Ram and Getz, Noah and Portnoi, Tally and Roy, Julien and Stark, Hannes and Kwabi-Addo, David and Beaini, Dominique and Jaakkola, Tommi and Barzilay, Regina},
  title = {Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction},
  year = {2025},
  doi = {10.1101/2025.06.14.659707},
  journal = {bioRxiv}
}

@article{wohlwend2024boltz1,
  author = {Wohlwend, Jeremy and Corso, Gabriele and Passaro, Saro and Getz, Noah and Reveiz, Mateo and Leidal, Ken and Swiderski, Wojtek and Atkinson, Liam and Portnoi, Tally and Chinn, Itamar and Silterra, Jacob and Jaakkola, Tommi and Barzilay, Regina},
  title = {Boltz-1: Democratizing Biomolecular Interaction Modeling},
  year = {2024},
  doi = {10.1101/2024.11.19.624167},
  journal = {bioRxiv}
}
```

### MMseqs2-GPU

```bibtex
@article{Kallenborn2025-fd,
  title     = "{GPU}-accelerated homology search with {MMseqs2}",
  author    = "Kallenborn, Felix and Chacon, Alejandro and Hundt, Christian and
               Sirelkhatim, Hassan and Didi, Kieran and Cha, Sooyoung and
               Dallago, Christian and Mirdita, Milot and Schmidt, Bertil and
               Steinegger, Martin",
  journal   = "Nat. Methods",
  volume    =  22,
  number    =  10,
  pages     = "2024--2027",
  year      =  2025,
  doi       = "10.1038/s41592-025-02819-8",
}
```

## License

MIT License. All code and model weights are freely available for both academic and commercial use.
