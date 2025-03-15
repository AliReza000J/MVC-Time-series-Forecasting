import tensorflow as tf
from tensorflow.keras import layers, Sequential, Input, Model

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