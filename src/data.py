from pathlib import Path

import pandas as pd
from scipy.io.arff import loadarff

from src.config import TARGET_COL


def load_data(path: str | Path, deduplicate: bool = True) -> pd.DataFrame:
    """Load the Polish companies bankruptcy dataset from an ARFF file.

    Expects the UCI format, where the target column `class` arrives as byte
    strings and is decoded here.

    Args:
        path: Path to the ARFF file.
        deduplicate: Whether to drop full-row duplicates, keeping the first
            occurrence. Defaults to True; pass False to inspect the raw file.

    Returns:
        Features under their original names (Attr1..Attr64) plus the binary
        target, with the index reset.
    """
    raw_data, _ = loadarff(path)
    df = pd.DataFrame(raw_data)

    df["class"] = df["class"].str.decode("utf-8").astype(int)

    df = df.rename(columns={"class": TARGET_COL})

    if deduplicate:
        df = df.drop_duplicates(keep="first")
    df = df.reset_index(drop=True)

    return df
