import pandas as pd


def add_negative_equity_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add a boolean flag marking rows where equity < 0."""
    result = df.copy()
    result["has_negative_equity"] = df["Attr10"] < 0
    return result
