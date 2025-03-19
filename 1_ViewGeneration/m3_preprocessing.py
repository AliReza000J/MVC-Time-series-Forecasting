import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

def load_dataset(file_path):
    """
    Load the M3 Monthly dataset and check if the file exists.
    """
    if os.path.exists(file_path):
        m3 = pd.read_csv(file_path)
        print("Dataset loaded successfully!")
        return m3
    else:
        print(f"Error: File '{file_path}' not found. Please check the path.")
        return None

def preprocessing(data, horizon=18):
    """
    Preprocess the time-series data:
    - Remove NaN values
    - Exclude last `horizon` values to prevent data leakage
    - Normalize using mean-variance scaling
    - Convert to suitable arrays for deep learning
    """
    ts_train = []
    for i in range(data.shape[0]):
        temp = np.array(list(data.iloc[i][6:].dropna())[:-horizon])  # Remove last `horizon` values
        temp = temp.reshape(1, len(temp), 1)
        temp = TimeSeriesScalerMeanVariance().fit_transform(temp)  # Normalize
        ts_train.append(temp.reshape(-1, 1))  # Reshape to (timesteps, 1)
    return ts_train

def process_m3_dataset(file_path):
    """
    Full processing pipeline for the M3 Monthly dataset:
    - Loads dataset
    - Groups by category
    - Preprocesses each category separately
    - Pads sequences to max length in each category
    - Returns reshaped data ready for deep learning models
    """
    m3 = load_dataset(file_path)
    if m3 is None:
        return None

    class_dataframes = {}  # Store separate DataFrames for each category
    processed_datasets = {}  # Store processed time-series
    max_seq_lengths = {}  # Store max sequence lengths per category
    padded_sequences = {}  # Store padded sequences
    reshaped_arrays = {}  # Store final reshaped arrays

    # Group time series data by 'Category' (e.g., MICRO, MACRO, INDUSTRY, etc.)
    for class_label in m3['Category'].unique():
        clean_label = class_label.replace(" ", "")  # Remove spaces for consistency
        class_dataframes[clean_label] = m3[m3['Category'] == class_label]

    # Preprocess each category separately
    for class_label, df in class_dataframes.items():
        processed_datasets[class_label] = preprocessing(df)
        max_seq_lengths[class_label] = max(len(seq) for seq in processed_datasets[class_label])

    # Pad sequences to max length per category
    for label, dataset in processed_datasets.items():
        padded_sequences[label] = tf.keras.preprocessing.sequence.pad_sequences(
            dataset, maxlen=max_seq_lengths[label], padding='post', dtype='float32'
        )

    # Reshape data to (batch_size, 1, timesteps) for LSTMs/CNNs
    for label, padded in padded_sequences.items():
        reshaped_arrays[label] = padded.reshape(padded.shape[0], 1, padded.shape[1])

    print("Data preprocessing completed successfully!")
    return reshaped_arrays