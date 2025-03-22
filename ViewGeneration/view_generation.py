import os
import pandas as pd
import numpy as np
from pyts.image import RecurrencePlot, GramianAngularField, MarkovTransitionField
import matplotlib.pyplot as plt
import pandas as pd

import pandas as pd
import numpy as np

import pandas as pd

import pandas as pd
from collections import OrderedDict

def process_m3_dataset(file_path, selected_category=None):
    """
    Processes the M3 dataset and extracts time-series data dynamically, preserving the original order of categories.

    Parameters:
    - file_path (str): Path to the M3 dataset CSV file.
    - selected_category (str, optional): The category of data to extract. If None, all categories are processed.

    Returns:
    - OrderedDict: An ordered dictionary containing processed DataFrames for each category.
    """
    # Load dataset
    data = pd.read_csv(file_path)

    # Preserve the original order of categories
    ordered_categories = list(data['Category'].unique())  
    splited_dataframes = OrderedDict((cat.strip(), data[data['Category'] == cat].copy()) for cat in ordered_categories)

    # If a specific category is requested, extract it
    if selected_category:
        selected_category = selected_category.strip()
        if selected_category in splited_dataframes:
            df = splited_dataframes[selected_category]
            df = df.drop(["Series", "N", "NF", "Category", "Starting Year", "Starting Month"], axis=1)
            return OrderedDict({selected_category: df})
        else:
            raise ValueError(f"Category '{selected_category}' not found in dataset.")
    
    # Process all categories while keeping the original order
    for key in splited_dataframes:
        splited_dataframes[key] = splited_dataframes[key].drop(
            ["Series", "N", "NF", "Category", "Starting Year", "Starting Month"], axis=1
        )

    return splited_dataframes

def preprocess_series(series, h):
    """Preprocess a single time series: drop NaN values and normalize."""
    series = series.dropna()[:-h]  # Remove last h values and NaNs
    series = pd.DataFrame(series)
    series = series.rename(columns={ series.columns[0]: "Values" })
    return series

def normalize_series(series):
    """Normalize the time series based on its mean value."""
    X = series.values.reshape(1, -1)
    X = X / np.mean(X)  # Normalize
    return np.array([X.flatten()])  # Flatten for transformation

def generate_rp_image(X):
    """Generate a Recurrence Plot (RP) representation."""
    rp = RecurrencePlot(threshold='distance')
    return rp.fit_transform(X)[0]

def generate_mtf_image(X):
    """Generate a Markov Transition Field (MTF) representation."""
    mtf = MarkovTransitionField()
    return mtf.fit_transform(X)[0]

def generate_gaf_image(X, method):
    """Generate a Gramian Angular Field (GAF) representation (summation/difference)."""
    gaf = GramianAngularField(method=method)
    return gaf.fit_transform(X)[0]

def plot_series(series, index, output_path):
    """Plot the time series and save the figure."""
    os.makedirs(f"{output_path}/ts", exist_ok=True)
    
    plt.figure(figsize=(10, 10))
    plt.plot(series["Values"], linestyle="solid")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.xticks(rotation=45)
    plt.savefig(f"{output_path}/ts/time-series{index}.png", bbox_inches="tight")
    plt.close()

def save_image(image, index, output_path, cmap):
    """Save an image representation of the time series."""
    os.makedirs(output_path, exist_ok=True)  # Ensure directory exists

    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap=cmap, origin="lower")
    plt.axis("off")
    plt.savefig(f"{output_path}/time-series{index}.png", bbox_inches="tight")
    plt.close()

def process_time_series(df, output_path, h=18, verbose=True, start_index=0):
    """Process all time series in the DataFrame."""
    os.makedirs(output_path, exist_ok=True)

    for i in range(len(df)):
        if verbose:
            print(f"Processing time series {start_index+i+1}/{len(df)+start_index}...")
        
        series = preprocess_series(df.iloc[i], h)
        plot_series(series, i, output_path)
        X = normalize_series(series)
        
        # Generate image representations
        X_rp = generate_rp_image(X)
        X_mtf = generate_mtf_image(X)
        X_gasf = generate_gaf_image(X, "summation")
        X_gadf = generate_gaf_image(X, "difference")

        # Save images
        save_image(X_rp, start_index+i, f"{output_path}/rp", "binary")
        save_image(X_mtf, start_index+i, f"{output_path}/mtf", "rainbow")
        save_image(X_gasf, start_index+i, f"{output_path}/gasf", "rainbow")
        save_image(X_gadf, start_index+i, f"{output_path}/gadf", "rainbow")

    if verbose:
        print("All time series processed successfully.")