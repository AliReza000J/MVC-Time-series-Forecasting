import os
import pandas as pd
import numpy as np

def get_m3_dataset_params(dataset_name, base_path="/kaggle/input/newtsdatasets"):
    """
    Retrieve dataset parameters for different M3 dataset categories.

    Args:
        dataset_name (str): The specific M3 dataset category (e.g., 'm3-demo', 'm3-finance').
        base_path (str): The base path where datasets are stored (default: Kaggle dataset path).

    Returns:
        tuple: A tuple containing (dataset, labels, lag, look_forward, sample_overlap, 
               learning_rate, dataset_path, suilin_smape, frequency)
    """
    
    # Define dataset-specific configurations
    dataset_configs = {
        'm3-demo':     {'file': 'm3-demo-new2.xlsx',  'lag': 28, 'look_forward': 18, 'batch_size': 7,  'epochs': 20},
        'm3-finance':  {'file': 'm3-finance-new.xlsx','lag': 28, 'look_forward': 18, 'batch_size': 9,  'epochs': 25},
        'm3-industry': {'file': 'm3-industry-new.xlsx','lag': 36, 'look_forward': 18, 'batch_size': 10, 'epochs': 25},
        'm3-macro':    {'file': 'm3-macro-new.xlsx', 'lag': 28, 'look_forward': 18, 'batch_size': 15, 'epochs': 25},
        'm3-micro':    {'file': 'M3-micro-new.xlsx', 'lag': 28, 'look_forward': 18, 'batch_size': 18, 'epochs': 25},
        'm3-other':    {'file': 'm3-other-new.xlsx', 'lag': 28, 'look_forward': 18, 'batch_size': 2,  'epochs': 25},
    }

    # Ensure dataset name is valid
    if dataset_name not in dataset_configs:
        raise ValueError(f"Invalid dataset name '{dataset_name}'. Choose from {list(dataset_configs.keys())}.")

    # Extract dataset settings
    config = dataset_configs[dataset_name]
    dataset_path = os.path.join(base_path, 'M3')
    file_path = os.path.join(dataset_path, config['file'])
    
    # Load dataset (Excel format)
    raw_data = pd.read_excel(file_path, header=None).to_numpy().astype('float64')

    # Load labels (assuming a standard path structure for labels)
    labels_file = f"{dataset_name}_fcm_raw_16_noFeWe_NR2.npy"
    labels_path = os.path.join("/kaggle/input/pure-normalized/Nofeatureweight/raw/16", labels_file)
    labels = np.load(labels_path)

    # Define training parameters
    lag = config['lag']
    look_forward = config['look_forward']
    batch_size = config['batch_size']
    epochs = config['epochs']
    learning_rate = 0.0001  # Fixed learning rate for M3 datasets
    frequency = 12  # M3 datasets typically have monthly frequency

    # Compute sample overlap for sliding window approach
    sample_overlap = look_forward - 1

    # Preprocess dataset: Remove NaNs and store time series lengths
    dataset = [ts[~np.isnan(ts)] for ts in raw_data]
    seri_len = [len(ts) for ts in dataset]

    print(f"Dataset: {dataset_name} | Min Length: {np.min(seri_len)} | Max Length: {np.max(seri_len)}")

    return dataset, labels, lag, look_forward, sample_overlap, learning_rate, dataset_path, False, frequency