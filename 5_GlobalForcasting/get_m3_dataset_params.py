import numpy as np
import pandas as pd

def get_m3_dataset(dataset_name, dataset_dir):
    """
    Loads the selected M3 dataset and returns processed data along with its parameters.

    Args:
        dataset_name (str): One of the M3 subcategories ('m3-demo', 'm3-finance', 'm3-industry', 
                            'm3-macro', 'm3-micro', 'm3-other').
        dataset_dir (str): Path to the directory containing the dataset files.

    Returns:
        tuple: (dataset, labels, lag, look_forward, sample_overlap, learning_rate, frequency)
    """

    m3_datasets = {
        'm3-demo': {'file': 'm3-demo-new2.xlsx', 'lag': 28, 'look_forward': 18, 'batch_size': 7, 'epochs': 20, 'lr': 0.0001, 'freq': 12},
        'm3-finance': {'file': 'm3-finance-new.xlsx', 'lag': 28, 'look_forward': 18, 'batch_size': 9, 'epochs': 25, 'lr': 0.0001, 'freq': 12},
        'm3-industry': {'file': 'm3-industry-new.xlsx', 'lag': 36, 'look_forward': 18, 'batch_size': 10, 'epochs': 25, 'lr': 0.0001, 'freq': 12},
        'm3-macro': {'file': 'm3-macro-new.xlsx', 'lag': 28, 'look_forward': 18, 'batch_size': 15, 'epochs': 25, 'lr': 0.0001, 'freq': 12},
        'm3-micro': {'file': 'M3-micro-new.xlsx', 'lag': 28, 'look_forward': 18, 'batch_size': 18, 'epochs': 25, 'lr': 0.0001, 'freq': None},
        'm3-other': {'file': 'm3-other-new.xlsx', 'lag': 28, 'look_forward': 18, 'batch_size': 2, 'epochs': 25, 'lr': 0.0001, 'freq': 12}
    }

    if dataset_name not in m3_datasets:
        raise ValueError(f"Invalid dataset name: {dataset_name}. Choose from {list(m3_datasets.keys())}")

    # Load dataset
    dataset_path = f"{dataset_dir}/{m3_datasets[dataset_name]['file']}"
    raw_data = pd.read_excel(dataset_path, header=None).to_numpy().astype('float64')

    # Load labels if available
    labels_path = f"{dataset_dir}/{dataset_name}_labels.npy"
    try:
        labels = np.load(labels_path)
    except FileNotFoundError:
        labels = None  # Handle missing labels

    # Process series (remove NaN values)
    dataset = [series[~np.isnan(series)] for series in raw_data]

    # Extract parameters
    lag = m3_datasets[dataset_name]['lag']
    look_forward = m3_datasets[dataset_name]['look_forward']
    sample_overlap = look_forward - 1
    learning_rate = m3_datasets[dataset_name]['lr']
    frequency = m3_datasets[dataset_name]['freq']

    print(f"Loaded {dataset_name}: min_length={min(map(len, dataset))}, max_length={max(map(len, dataset))}")

    return dataset, labels, lag, look_forward, sample_overlap, learning_rate, frequency