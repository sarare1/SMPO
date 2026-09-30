"""Download the AI4I 2020 and Microsoft Azure Predictive Maintenance datasets.

Usage:  python -m src.data.download
"""
from __future__ import annotations

import io
import urllib.request
import zipfile

from src import config


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=300) as resp:
        return resp.read()


def download_ai4i(force: bool = False) -> None:
    target = config.DATA_RAW / "ai4i2020.csv"
    if target.exists() and not force:
        print(f"[skip] {target.name} exists")
        return
    print("Downloading AI4I 2020 from UCI ...")
    with zipfile.ZipFile(io.BytesIO(_fetch(config.AI4I_URL))) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        target.write_bytes(zf.read(name))
    print(f"[ok] {target}")


def download_azure_pdm(force: bool = False) -> None:
    for name in config.AZURE_PDM_FILES:
        target = config.DATA_RAW / f"{name}.csv"
        if target.exists() and not force:
            print(f"[skip] {target.name} exists")
            continue
        print(f"Downloading {name}.csv ...")
        target.write_bytes(_fetch(f"{config.AZURE_PDM_BASE}/{name}.csv"))
        print(f"[ok] {target}")


def main() -> None:
    download_ai4i()
    download_azure_pdm()


if __name__ == "__main__":
    main()
