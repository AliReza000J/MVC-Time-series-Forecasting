import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras
from PIL import Image
from keras import layers, Sequential, Input, Model


# ================================
#  Base Image Autoencoder
# ================================

class ImageAE:
    def __init__(self, latent_dim=32, img_input_shape=(64, 64, 3), filters=[32, 64, 128]):
        """
        Image-based Autoencoder with an Encoder-Decoder structure.

        Parameters:
        - latent_dim: Size of the latent space (default=32)
        - img_input_shape: Shape of input images (default=(64,64,3))
        - filters: List of filter sizes for convolutional layers (default=[32, 64, 128])
        """
        self.latent_dim = latent_dim
        self.img_input_shape = img_input_shape
        self.filters = filters

        # Build models
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.autoencoder = self.build_autoencoder()

    def build_encoder(self):
        """Builds the encoder model."""
        encoder = Sequential(name="Encoder")
        encoder.add(Input(shape=self.img_input_shape))
        
        # Convolutional layers with BatchNorm and Dropout
        for f in self.filters:
            encoder.add(layers.Conv2D(f, (3, 3), activation="relu", padding="same"))
            encoder.add(layers.BatchNormalization())
            encoder.add(layers.MaxPooling2D((2, 2)))
            encoder.add(layers.Dropout(0.2))  # Reduce overfitting
        
        # Latent representation
        encoder.add(layers.Flatten())
        encoder.add(layers.Dense(self.latent_dim, activation=layers.LeakyReLU(alpha=0.1)))
        return encoder

    def build_decoder(self):
        """Builds the decoder model."""
        decoder = Sequential(name="Decoder")
        decoder.add(Input(shape=(self.latent_dim,)))

        # Fully connected layer before reshaping
        decoder.add(layers.Dense(8 * 8 * self.filters[-1], activation="relu"))
        decoder.add(layers.Reshape((8, 8, self.filters[-1])))

        # Upsampling + Transposed Convolutions
        for f in reversed(self.filters[:-1]):
            decoder.add(layers.Conv2DTranspose(f, (3, 3), strides=2, activation="relu", padding="same"))
            decoder.add(layers.BatchNormalization())

        decoder.add(layers.Conv2DTranspose(self.img_input_shape[2], (3, 3), activation="sigmoid", padding="same"))
        return decoder

    def build_autoencoder(self):
        """Combines the encoder and decoder into a full autoencoder."""
        input_img = Input(shape=self.img_input_shape)
        encoded_img = self.encoder(input_img)
        decoded_img = self.decoder(encoded_img)
        return Model(inputs=input_img, outputs=decoded_img, name="ImageAutoencoder")

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
    model.add(keras.layers.Masking(mask_value=0.0))

    # Encoder
    for i, units in enumerate(encoder_hiddens[:-1]):
        model.add(tf.keras.layers.LSTM(units, return_sequences=True))
    model.add(tf.keras.layers.LSTM(encoder_hiddens[-1]))  # Last layer without return_sequences

    # Latent Space
    model.add(keras.layers.Dense(latent_dim, name="emb"))

    # Decoder
    model.add(keras.layers.RepeatVector(1))
    for units in decoder_hiddens:
        model.add(tf.keras.layers.LSTM(units, return_sequences=True))
    
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(series_len)))  # Output layer

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