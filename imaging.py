import pandas as pd
import numpy as np
from pyts.image import RecurrencePlot
from pyts.image import GramianAngularField
from pyts.image import MarkovTransitionField
import matplotlib.pyplot as plt

def preprocess_series(series, h):
    """Preprocess a single time series: drop NaN values and normalize."""
    series = series.dropna()[:-h]  # Remove last h values and NaNs
    series = pd.DataFrame(series, columns=["Values"])
    return series

def normalize_series(series):
    """Normalize the time series based on its mean value."""
    X = series.values.reshape(1, -1)
    X = X / np.mean(X)  # Normalize
    return np.array([X.flatten()])  # Flatten for transformation

def generate_rp_image(X):
    """Generate a Recurrence Plot (RP) representation."""
    rp = RecurrencePlot(threshold='distance')
    X_rp = rp.fit_transform(X)
    return X_rp[0]

def generate_mtf_image(X):
    """Generate a Markov Transition Field (MTF) representation."""
    mtf = MarkovTransitionField()
    X_mtf = mtf.fit_transform(X)
    return X_mtf[0]

def generate_gaf_image(X, method):
    """Generate a Gramian Angular Field (GAF) representation."""
    gaf = GramianAngularField(method=method)
    X_gaf = gaf.fit_transform(X)
    return X_gaf[0]

def plot_series(series, index, output_path):
    """Plot the time series and save the figure."""
    plt.figure(figsize=(10, 10))
    plt.plot(series["Values"], linestyle='solid')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.savefig(f"{output_path}time-series{index}.png", bbox_inches="tight")
    plt.close()

def save_image(image, index, output_path, cmap):
    """Save the image with the given index."""
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap=cmap, origin="lower")
    plt.axis("off")
    plt.savefig(f"{output_path}time-series{index}.png", bbox_inches="tight")
    plt.close()

def process_time_series(df, output_path, h=18):
    """Process all time series in the DataFrame."""
    for i in range(len(df)):
        series = preprocess_series(df.iloc[i], h)
        plot_series(series, i, output_path)
        X = normalize_series(series)
        
        # Generate images
        X_rp = generate_rp_image(X)
        X_mtf = generate_mtf_image(X)
        X_gasf = generate_gaf_image(X, 'summation')
        X_gadf = generate_gaf_image(X, 'difference')

        # Save images
        save_image(X_rp, i, output_path, 'binary')
        save_image(X_mtf, i, output_path, "rainbow")
        save_image(X_gasf, i, output_path, "rainbow")
        save_image(X_gadf, i, output_path, "rainbow")