import os
import re

import requests


def parse_benchmark_lines(lines: list[str]) -> dict:
    """Parse [Benchmark] lines from boltz predict stdout.

    Parameters
    ----------
    lines : list[str]
        Lines of stdout output from boltz predict.

    Returns
    -------
    dict
        Parsed benchmark values (msa_time_s, inference_time_s, total_time_s,
        num_inputs). Values are None if not found.

    """
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


def download_file(url, filepath, verbose=True):
    if verbose:
        print(f"Downloading {url} to {filepath}")
    response = requests.get(url)

    target_dir = os.path.dirname(filepath)
    if target_dir and not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Check if the request was successful
    if response.status_code == 200:
        with open(filepath, "wb") as file:
            file.write(response.content)
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

    return filepath
