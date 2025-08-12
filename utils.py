# utils.py
import os
import pandas as pd

def load_data(client_name, base_dir=None):
    """
    Loads the CSV for a given client from sample_data/.
    Example: load_data("bank") -> loads sample_data/bank.csv

    Parameters:
    client_name (str): One of 'aadhaar_office','bank','insurance','pension',
                       'telecom','death_registrar','tax'
    base_dir (str): Path to project root (defaults to script's dir)

    Returns:
    pd.DataFrame: Client's CSV as a DataFrame
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    # Data folder and file path
    data_dir = os.path.join(base_dir, "sample_data")
    file_path = os.path.join(data_dir, f"{client_name}.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file for '{client_name}' not found: {file_path}")

    df = pd.read_csv(file_path)
    return df
