import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Masking
import tensorflow_addons as tfa
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import keras.backend as K
import gc
from PIL import Image
from sklearn_extra.cluster import KMedoids
import ImageAE 
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import numpy as np
import pandas as pd


# ================================
#  Prepare Data for Extracting Features
# ================================

def load_dataset(file_path):
    """Load the dataset and check if the file exists."""
    if os.path.exists(file_path):
        m3 = pd.read_csv(file_path)
        print("✅ Dataset loaded successfully!")
        return m3
    else:
        print(f"❌ Error: File '{file_path}' not found. Please check the path.")
        return None

def preprocessing(data, horizon=18):
    """
    Preprocess time series data:
    - Remove `NaN` values
    - Normalize using mean-variance scaling
    - Convert to arrays suitable for deep learning
    """
    ts_train = []
    for i in range(data.shape[0]):
        temp = np.array(list(data.iloc[i][6:].dropna())[:-horizon])  # Remove forecasting horizon
        temp = temp.reshape(1, len(temp), 1)
        temp = TimeSeriesScalerMeanVariance().fit_transform(temp)
        ts_train.append(temp.reshape(-1, 1))
    return ts_train

def process_m3_dataset(file_path):
    """Complete processing pipeline for the M3 dataset."""
    m3 = load_dataset(file_path)
    if m3 is None:
        return None

    class_dataframes = {}  # Store DataFrames for each category
    processed_datasets = {}  # Store preprocessed data
    max_seq_lengths = {}  # Max sequence length per category
    padded_sequences = {}  # Store padded data
    reshaped_arrays = {}  # Store final reshaped data

    # Categorize data based on series type
    for class_label in m3['Category'].unique():
        clean_label = class_label.replace(" ", "")
        class_dataframes[clean_label] = m3[m3['Category'] == class_label]

    # Process time series for each category
    for class_label, df in class_dataframes.items():
        processed_datasets[class_label] = preprocessing(df)
        max_seq_lengths[class_label] = max(len(seq) for seq in processed_datasets[class_label])

    # Pad sequences to ensure uniform length
    for label, dataset in processed_datasets.items():
        padded_sequences[label] = tf.keras.preprocessing.sequence.pad_sequences(
            dataset, maxlen=max_seq_lengths[label], padding='post', dtype='float32'
        )

    # Reshape data for deep learning models
    for label, padded in padded_sequences.items():
        reshaped_arrays[label] = padded.reshape(padded.shape[0], 1, padded.shape[1])

    print("Data preprocessing completed successfully!")
    return reshaped_arrays

# ================================
#  Autoencoder for Time-Series Data
# ================================

def build_time_series_autoencoder(latent_dim=8, encoder_hiddens=[256, 128, 64], 
                                  decoder_hiddens=[64, 128, 256], series_len=None):
    """
    Builds an LSTM-based autoencoder for time-series feature extraction.
    
    Args:
        latent_dim (int): Dimensionality of the latent space.
        encoder_hiddens (list): List of hidden units in the encoder LSTM layers.
        decoder_hiddens (list): List of hidden units in the decoder LSTM layers.
        series_len (int): Length of the time series.
    
    Returns:
        keras.Model: Compiled autoencoder model.
    """
    model = tf.keras.Sequential()
    model.add(keras.Input((None, series_len)))  # Input shape (batch_size, timesteps, series_length)
    
    # Masking to ignore padding values (if any)
    model.add(Masking(mask_value=0.0))

    # Encoder
    for i, units in enumerate(encoder_hiddens[:-1]):
        model.add(LSTM(units, return_sequences=True))
    model.add(LSTM(encoder_hiddens[-1]))  # Last layer without return_sequences

    # Latent Space
    model.add(Dense(latent_dim, name="emb"))

    # Decoder
    model.add(RepeatVector(1))
    for units in decoder_hiddens:
        model.add(LSTM(units, return_sequences=True))
    
    model.add(TimeDistributed(Dense(series_len)))  # Output layer

    return model


# ================================
#  Image Preprocessing
# ================================

def load_and_preprocess_images(image_paths, image_size=(64, 64), gray=True):
    """
    Loads and preprocesses images for model input.
    
    Args:
        image_paths (list): List of image file paths.
        image_size (tuple): Target size for resizing images.
        gray (bool): Whether to convert images to grayscale.
    
    Returns:
        np.array: Preprocessed image array.
    """
    images = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('L' if gray else "RGB")
        img = img.resize(image_size)
        img_array = np.array(img) / 255.0  # Normalize to [0,1]
        images.append(img_array)
    
    return np.array(images)


# ================================
#  Image Feature Extraction
# ================================

def extract_image_features(reshaped_array, image_dir, gray=True):
    """
    Extracts features from images using an autoencoder.
    
    Args:
        reshaped_array (np.array): Time-series data for alignment.
        image_dir (str): Directory containing image files.
        gray (bool): Whether images are grayscale or RGB.
    
    Returns:
        np.array: Extracted image features.
    """
    image_size = (64, 64)
    image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir)]
    
    # Load and preprocess images
    image_data = load_and_preprocess_images(image_paths, image_size, gray)
    print("Image data shape:", image_data.shape)

    # Ensure consistency in dataset size
    assert image_data.shape[0] == reshaped_array.shape[0], "Mismatch in data sizes!"

    # Initialize Image Autoencoder
    image_ae = ImageAE(latent_dim=8, img_input_shape=(64, 64, 1 if gray else 3))
    autoencoder_model = image_ae.autoencoder

    # Compile model
    autoencoder_model.compile(optimizer="adam", loss="mse")

    # Train autoencoder
    history = autoencoder_model.fit(image_data, image_data, epochs=100, batch_size=100)

    # Print training history
    print("Training loss history:", history.history["loss"])

    # Extract features using encoder
    encoder_model = image_ae.encoder
    features_img = encoder_model.predict(image_data)

    print("Extracted feature shape:", features_img.shape)
    return features_img


# ================================
#  Time-Series Feature Extraction
# ================================

def extract_time_series_features(reshaped_array):
    """
    Extracts features from time-series data using an LSTM autoencoder.
    
    Args:
        reshaped_array (np.array): Time-series data.
    
    Returns:
        np.array: Extracted time-series features.
    """
    latent_dim = 8
    series_len = reshaped_array.shape[2]

    # Initialize and compile autoencoder
    ts_ae = build_time_series_autoencoder(series_len=series_len)
    ts_ae.compile(optimizer=keras.optimizers.Adam(), loss="mse")

    # Train autoencoder
    ts_ae.fit(reshaped_array, reshaped_array, epochs=100, batch_size=10, verbose=1)

    # Extract features
    encoder_ts_ae = keras.Model(ts_ae.input, ts_ae.get_layer('emb').output)
    features_ts_ae = encoder_ts_ae.predict(reshaped_array)

    return features_ts_ae


# # ================================
# #  Main Function
# # ================================
# if __name__ == "__main__":
#     file_path = "m3monthdataset/M3Month.csv"
#     processed_data = process_m3_dataset(file_path)
#     if processed_data:
#         print("Available categories:", list(processed_data.keys()))


# if __name__ == "__main__":
#     """
#     Main execution script. Load data, process images & time-series, extract features.
#     """
#     # Example usage (paths should be updated based on your dataset)
#     time_series_data = np.load("your_time_series_data.npy")  # Update with actual path
#     image_directory = "your_image_directory/"  # Update with actual path
    
#     # Extract features
#     ts_features = extract_time_series_features(time_series_data)
#     img_features = extract_image_features(time_series_data, image_directory, gray=True)

#     # Save extracted features
#     np.save("time_series_features.npy", ts_features)
#     np.save("image_features.npy", img_features)

#     print("Feature extraction complete. Files saved.")