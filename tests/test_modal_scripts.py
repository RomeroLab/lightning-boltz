"""Tests for modal/ scripts.

Tests parsing logic, CLI structure, and shared constants.
No Modal or GPU required.
"""

import re
from pathlib import Path

MODAL_DIR = Path("modal")


# ---------------------------------------------------------------------------
# Helper: replicate benchmark parsing from benchmark.py
# ---------------------------------------------------------------------------
def parse_benchmark_lines(lines: list[str]) -> dict:
    """Parse [Benchmark] lines from boltz predict stdout."""
    msa_time = None
    inference_time = None
    total_time = None
    num_inputs = None

    for line in lines:
        if "[Benchmark] MSA generation + preprocessing:" in line:
            m = re.search(r"([\d.]+)s", line)
            if m:
                msa_time = float(m.group(1))
        elif "[Benchmark] Inference:" in line:
            m = re.search(r"([\d.]+)s", line)
            if m:
                inference_time = float(m.group(1))
        elif "[Benchmark] Total:" in line:
            m = re.search(r"([\d.]+)s", line)
            if m:
                total_time = float(m.group(1))
        elif "[Benchmark] Num inputs:" in line:
            m = re.search(r"(\d+)", line.split("Num inputs:")[-1])
            if m:
                num_inputs = int(m.group(1))

    return {
        "msa_time_s": msa_time,
        "inference_time_s": inference_time,
        "total_time_s": total_time,
        "num_inputs": num_inputs,
    }


# ---------------------------------------------------------------------------
# Benchmark parsing tests
# ---------------------------------------------------------------------------
class TestBenchmarkParsing:
    """Test parsing of [Benchmark] output lines."""

    def test_parse_typical_output(self):
        lines = [
            "Processing inputs...\n",
            "\n",
            "[Benchmark] MSA generation + preprocessing: 42.3s\n",
            "Running structure prediction for 10 inputs.\n",
            "\n",
            "[Benchmark] Inference: 120.5s\n",
            "[Benchmark] MSA generation + preprocessing: 42.3s\n",
            "[Benchmark] Inference: 120.5s\n",
            "[Benchmark] Total: 162.8s\n",
            "[Benchmark] Num inputs: 10\n",
        ]
        result = parse_benchmark_lines(lines)
        assert result["msa_time_s"] == 42.3
        assert result["inference_time_s"] == 120.5
        assert result["total_time_s"] == 162.8
        assert result["num_inputs"] == 10

    def test_parse_no_benchmark_lines(self):
        lines = ["Some other output\n", "No benchmark here\n"]
        result = parse_benchmark_lines(lines)
        assert result["msa_time_s"] is None
        assert result["inference_time_s"] is None
        assert result["total_time_s"] is None
        assert result["num_inputs"] is None

    def test_parse_zero_times(self):
        lines = [
            "[Benchmark] MSA generation + preprocessing: 0.0s\n",
            "[Benchmark] Inference: 0.0s\n",
            "[Benchmark] Total: 0.0s\n",
            "[Benchmark] Num inputs: 0\n",
        ]
        result = parse_benchmark_lines(lines)
        assert result["msa_time_s"] == 0.0
        assert result["inference_time_s"] == 0.0
        assert result["total_time_s"] == 0.0
        assert result["num_inputs"] == 0

    def test_parse_large_times(self):
        lines = [
            "[Benchmark] MSA generation + preprocessing: 3600.5s\n",
            "[Benchmark] Inference: 7200.1s\n",
            "[Benchmark] Total: 10800.6s\n",
            "[Benchmark] Num inputs: 512\n",
        ]
        result = parse_benchmark_lines(lines)
        assert result["msa_time_s"] == 3600.5
        assert result["inference_time_s"] == 7200.1
        assert result["total_time_s"] == 10800.6
        assert result["num_inputs"] == 512

    def test_parse_single_input(self):
        lines = [
            "[Benchmark] MSA generation + preprocessing: 5.2s\n",
            "[Benchmark] Inference: 8.7s\n",
            "[Benchmark] Total: 13.9s\n",
            "[Benchmark] Num inputs: 1\n",
        ]
        result = parse_benchmark_lines(lines)
        assert result["num_inputs"] == 1

    def test_parse_partial_output(self):
        lines = ["[Benchmark] MSA generation + preprocessing: 42.3s\n"]
        result = parse_benchmark_lines(lines)
        assert result["msa_time_s"] == 42.3
        assert result["inference_time_s"] is None
        assert result["total_time_s"] is None

    def test_parse_duplicate_lines_takes_last(self):
        lines = [
            "[Benchmark] MSA generation + preprocessing: 42.3s\n",
            "[Benchmark] Inference: 120.5s\n",
            "[Benchmark] MSA generation + preprocessing: 42.3s\n",
            "[Benchmark] Inference: 120.5s\n",
            "[Benchmark] Total: 162.8s\n",
        ]
        result = parse_benchmark_lines(lines)
        assert result["msa_time_s"] == 42.3
        assert result["inference_time_s"] == 120.5

    def test_parse_empty_lines(self):
        result = parse_benchmark_lines([])
        assert result["msa_time_s"] is None
        assert result["inference_time_s"] is None
        assert result["total_time_s"] is None
        assert result["num_inputs"] is None


# ---------------------------------------------------------------------------
# Script structure tests
# ---------------------------------------------------------------------------
class TestModalDirectory:
    """Test that modal/ directory has the expected scripts."""

    def test_directory_exists(self):
        assert MODAL_DIR.is_dir()

    def test_benchmark_script_exists(self):
        assert (MODAL_DIR / "benchmark.py").is_file()

    def test_upload_dbs_script_exists(self):
        assert (MODAL_DIR / "upload_dbs.py").is_file()

    def test_upload_models_script_exists(self):
        assert (MODAL_DIR / "upload_models.py").is_file()


class TestBenchmarkScript:
    """Test benchmark.py structure."""

    def test_has_expected_functions(self):
        content = (MODAL_DIR / "benchmark.py").read_text()
        assert "def setup_databases" in content
        assert "def download_checkpoints" in content
        assert "def run_benchmark" in content
        assert "def check_status" in content
        assert "@app.local_entrypoint()" in content

    def test_has_volume_names(self):
        content = (MODAL_DIR / "benchmark.py").read_text()
        assert "boltz-mmseqs-dbs" in content
        assert "boltz-cache" in content

    def test_has_gpu_spec(self):
        content = (MODAL_DIR / "benchmark.py").read_text()
        assert 'gpu="A100-80GB"' in content

    def test_has_two_stage_db_setup(self):
        content = (MODAL_DIR / "benchmark.py").read_text()
        assert "staging" in content.lower()

    def test_raises_on_error(self):
        content = (MODAL_DIR / "benchmark.py").read_text()
        assert "raise RuntimeError" in content

    def test_filters_files_only_in_copy(self):
        """Verify setup_databases filters for is_file() during copy."""
        content = (MODAL_DIR / "benchmark.py").read_text()
        assert "if f.is_file()" in content or "f.is_file()" in content

    def test_handles_symlinks_in_copy(self):
        """Verify setup_databases copies symlinks separately."""
        content = (MODAL_DIR / "benchmark.py").read_text()
        assert "is_symlink" in content


class TestUploadDbsScript:
    """Test upload_dbs.py structure."""

    def test_has_setup_function(self):
        content = (MODAL_DIR / "upload_dbs.py").read_text()
        assert "def setup_databases" in content

    def test_has_hf_download_function(self):
        content = (MODAL_DIR / "upload_dbs.py").read_text()
        assert "def download_from_hf" in content

    def test_uses_correct_volume(self):
        content = (MODAL_DIR / "upload_dbs.py").read_text()
        assert "boltz-mmseqs-dbs" in content

    def test_has_local_entrypoint(self):
        content = (MODAL_DIR / "upload_dbs.py").read_text()
        assert "@app.local_entrypoint()" in content

    def test_runs_setup_script(self):
        content = (MODAL_DIR / "upload_dbs.py").read_text()
        assert "setup_boltz_mmseqs_dbs.sh" in content

    def test_has_two_stage_approach(self):
        content = (MODAL_DIR / "upload_dbs.py").read_text()
        assert "staging" in content.lower()
        assert "ephemeral" in content.lower()

    def test_supports_colabfold_and_alphafold3(self):
        content = (MODAL_DIR / "upload_dbs.py").read_text()
        assert "colabfold" in content
        assert "alphafold3" in content

    def test_has_from_hf_flag(self):
        content = (MODAL_DIR / "upload_dbs.py").read_text()
        assert "from_hf" in content
        assert "hf_repo" in content

    def test_has_default_hf_repo(self):
        content = (MODAL_DIR / "upload_dbs.py").read_text()
        assert "DEFAULT_HF_REPO" in content

    def test_hf_handles_split_parts(self):
        content = (MODAL_DIR / "upload_dbs.py").read_text()
        assert ".part" in content
        assert "reassembl" in content.lower()

    def test_raises_on_error(self):
        content = (MODAL_DIR / "upload_dbs.py").read_text()
        assert "raise RuntimeError" in content


class TestUploadModelsScript:
    """Test upload_models.py structure."""

    def test_has_upload_function(self):
        content = (MODAL_DIR / "upload_models.py").read_text()
        assert "def upload" in content

    def test_uses_correct_volume(self):
        content = (MODAL_DIR / "upload_models.py").read_text()
        assert "boltz-cache" in content

    def test_has_local_entrypoint(self):
        content = (MODAL_DIR / "upload_models.py").read_text()
        assert "@app.local_entrypoint()" in content

    def test_handles_expected_files(self):
        content = (MODAL_DIR / "upload_models.py").read_text()
        assert "boltz2_conf.ckpt" in content
        assert "mols" in content
