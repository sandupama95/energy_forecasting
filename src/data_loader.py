# src/data_loader.py

import pandas as pd

def load_raw_data(path: str):
    """
    Load CSV from 'path', parse 'Date' column as datetime, sort by Date.
    Returns: pd.DataFrame
    """
    df = pd.read_csv(path, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
