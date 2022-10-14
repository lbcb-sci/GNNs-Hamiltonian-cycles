from pathlib import Path

from hamgnn.constants import FHCP_BENCHMARK_DIR, FHCP_HOMEPAGE, FHCP_GRAPHS_URL, FHCP_SOLUTIONS_URL
import subprocess


if __name__ == "__main__":
    fhcp_data_dir = Path(FHCP_BENCHMARK_DIR)
    if not fhcp_data_dir.exists():
        fhcp_data_dir.mkdir()
    assert fhcp_data_dir.is_dir(), f"Wanted to create directory {FHCP_BENCHMARK_DIR} but a file with the same name already exists!"
    if len([p for p in fhcp_data_dir.iterdir()]) > 0:
        print(f"Found existing HCP instances in {fhcp_data_dir}. Aborting")
    else:
        print(f"Did not find any HCP instances, downloading from {FHCP_HOMEPAGE}")
        subprocess.call(["wget", FHCP_GRAPHS_URL], cwd=fhcp_data_dir)
        subprocess.call(["wget", FHCP_SOLUTIONS_URL], cwd=fhcp_data_dir)
        subprocess.call(["7z", "e", "."], cwd=fhcp_data_dir)
        print("Successfully downloaded HCP data into{TSPLIB_HCP_BENCHMARK}}")
