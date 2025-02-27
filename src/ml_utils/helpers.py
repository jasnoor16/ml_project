"""
This script contains helper functions used across the project.

It provides:
- Directory creation to ensure files can be saved correctly.
- CSV file loading into pandas DataFrames.
- Saving and loading NumPy arrays.
"""

import os
import numpy as np
import pandas as pd

def ensure_directory_exists(directory):
    """
    Ensures the specified directory exists.
    
    If the directory does not exist, it creates it.
    This is useful to avoid errors when trying to save files.

    Parameters:
    - directory (str): The directory path to be checked or created.

    Returns:
    - None
    """
    os.makedirs(directory, exist_ok=True)

def load_data(file_path):
    """
    Loads a CSV file into a pandas DataFrame.

    This function checks if the file exists before attempting to read it.
    If the file is missing, an error is raised.

    Parameters:
    - file_path (str): The full path to the CSV file.

    Returns:
    - pd.DataFrame: The loaded dataset.

    Raises:
    - FileNotFoundError: If the specified file does not exist.
    """
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        raise FileNotFoundError(f"File not found: {file_path}")

def save_numpy_array(array, file_path):
    """
    Saves a NumPy array as a .npy file.

    NumPy arrays are stored efficiently in binary format using .npy files.
    This function ensures the file is saved at the given location.

    Parameters:
    - array (np.ndarray): The NumPy array to be saved.
    - file_path (str): The path where the .npy file will be stored.

    Returns:
    - None
    """
    np.save(file_path, array)

def load_numpy_array(file_path):
    """
    Loads a .npy file into a NumPy array.

    This function checks if the file exists before loading it.
    If the file is missing, an error is raised.

    Parameters:
    - file_path (str): The full path to the .npy file.

    Returns:
    - np.ndarray: The loaded NumPy array.

    Raises:
    - FileNotFoundError: If the specified file does not exist.
    """
    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        raise FileNotFoundError(f"File not found: {file_path}")
