import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras
from PIL import Image
from tensorflow.keras import layers, Model, Input


# ================================
#  Base Image Autoencoder
# ================================

from tensorflow.keras import layers, Model, Input

class ImageAE:
    def __init__(self, latent_dim=32, img_input_shape=(64, 64, 3), filters=[32, 64, 128]):
        """
        Functional-style CNN Autoencoder for images.
        
        Args:
            latent_dim (int): Size of the latent space.
            img_input_shape (tuple): Shape of the input image.
            filters (list): List of filters per conv block.
        """
        self.latent_dim = latent_dim
        self.img_input_shape = img_input_shape
        self.filters = filters

        # Build the components
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.autoencoder = self.build_autoencoder()

    def build_encoder(self):
        input_img = Input(shape=self.img_input_shape, name="image_input")
        x = input_img

        # Convolutional encoder
        for i, f in enumerate(self.filters):
            x = layers.Conv2D(f, (3, 3), padding="same", activation="relu", name=f"enc_conv_{i}")(x)
            x = layers.BatchNormalization(name=f"enc_bn_{i}")(x)
            x = layers.MaxPooling2D((2, 2), name=f"enc_pool_{i}")(x)
            x = layers.Dropout(0.2, name=f"enc_dropout_{i}")(x)

        x = layers.Flatten(name="enc_flatten")(x)
        encoded = layers.Dense(self.latent_dim, activation=layers.LeakyReLU(alpha=0.1), name="latent")(x)

        return Model(inputs=input_img, outputs=encoded, name="ImageEncoder")

    def build_decoder(self):
        latent_input = Input(shape=(self.latent_dim,), name="latent_input")
        x = layers.Dense(8 * 8 * self.filters[-1], activation="relu", name="dec_dense")(latent_input)
        x = layers.Reshape((8, 8, self.filters[-1]), name="dec_reshape")(x)

        # Upsampling 3 times to get back to 64x64
        for i, f in enumerate(reversed(self.filters)):
            x = layers.Conv2DTranspose(f, (3, 3), strides=2, padding="same", activation="relu", name=f"dec_deconv_{i}")(x)
            x = layers.BatchNormalization(name=f"dec_bn_{i}")(x)

        x = layers.Conv2DTranspose(self.img_input_shape[2], (3, 3), activation="sigmoid", padding="same", name="dec_output")(x)

        return Model(inputs=latent_input, outputs=x, name="ImageDecoder")

    def build_autoencoder(self):
        input_img = Input(shape=self.img_input_shape, name="ae_input")
        encoded = self.encoder(input_img)
        decoded = self.decoder(encoded)
        return Model(inputs=input_img, outputs=decoded, name="ImageAutoencoder")

# ================================
#  Autoencoder for Time-Series Data
# ================================

def build_time_series_autoencoder(latent_dim=8, encoder_hiddens=[256, 128, 64], 
                                  decoder_hiddens=[64, 128, 256], series_len=None):
    """
    Builds an LSTM-based autoencoder for time-series feature extraction using the functional API.

    Args:
        latent_dim (int): Dimensionality of the latent space.
        encoder_hiddens (list): Hidden units for encoder LSTM layers.
        decoder_hiddens (list): Hidden units for decoder LSTM layers.
        series_len (int): Length of time series input.

    Returns:
        keras.Model: Autoencoder model.
    """
    input_layer = keras.Input(shape=(None, series_len), name="input")
    x = layers.Masking(mask_value=0.0)(input_layer)

    # Encoder
    for i, units in enumerate(encoder_hiddens[:-1]):
        x = layers.LSTM(units, return_sequences=True, name=f"enc_lstm_{i}")(x)
    x = layers.LSTM(encoder_hiddens[-1], name="enc_lstm_final")(x)

    # Latent representation
    encoded = layers.Dense(latent_dim, activation=None, name="emb")(x)

    # Decoder
    x = layers.RepeatVector(1, name="repeat_vector")(encoded)
    for i, units in enumerate(decoder_hiddens):
        x = layers.LSTM(units, return_sequences=True, name=f"dec_lstm_{i}")(x)

    output = layers.TimeDistributed(layers.Dense(series_len), name="output")(x)

    model = keras.Model(inputs=input_layer, outputs=output, name="TimeSeriesAutoencoder")
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