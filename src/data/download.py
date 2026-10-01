"""Download the AI4I 2020 and Microsoft Azure Predictive Maintenance datasets.

WHAT THIS FILE DOES (plain English)
-----------------------------------
Step 1 of the pipeline. It fetches the two public datasets from the internet and
saves them as CSV (spreadsheet-like) files in data/raw/:

  * AI4I 2020        - one zip file from the UCI repository, unpacked to ai4i2020.csv.
  * Azure PdM        - five CSV files (sensors, errors, maintenance, failures, machines).

Files that are already downloaded are skipped, so running it again is quick.

Usage:  python -m src.data.download
"""
# Lets Python understand modern type hints on all versions.
from __future__ import annotations

# --- Imports: tools this file needs -----------------------------------------
import io              # treats downloaded bytes like a file in memory
import urllib.request  # downloads files from a web address
import zipfile         # opens .zip archives

from src import config  # project settings: download links and target folders


def _fetch(url: str) -> bytes:
    """Download the file at `url` and return its raw content (waits up to 5 minutes)."""
    with urllib.request.urlopen(url, timeout=300) as resp:
        return resp.read()


def download_ai4i(force: bool = False) -> None:
    """Download the AI4I dataset (a zip) and save the CSV inside it to data/raw/ai4i2020.csv."""
    target = config.DATA_RAW / "ai4i2020.csv"
    # Skip the download if the file is already there (unless force=True).
    if target.exists() and not force:
        print(f"[skip] {target.name} exists")
        return
    print("Downloading AI4I 2020 from UCI ...")
    # Open the downloaded zip in memory, find the CSV file inside, and save it.
    with zipfile.ZipFile(io.BytesIO(_fetch(config.AI4I_URL))) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        target.write_bytes(zf.read(name))
    print(f"[ok] {target}")


def download_azure_pdm(force: bool = False) -> None:
    """Download the five Azure predictive-maintenance CSV files into data/raw/."""
    for name in config.AZURE_PDM_FILES:
        target = config.DATA_RAW / f"{name}.csv"
        # Skip files that are already downloaded.
        if target.exists() and not force:
            print(f"[skip] {target.name} exists")
            continue
        print(f"Downloading {name}.csv ...")
        target.write_bytes(_fetch(f"{config.AZURE_PDM_BASE}/{name}.csv"))
        print(f"[ok] {target}")


def main() -> None:
    """Download both datasets."""
    download_ai4i()
    download_azure_pdm()


# Runs only when this file is started directly (python -m src.data.download),
# not when another file imports it.
if __name__ == "__main__":
    main()
