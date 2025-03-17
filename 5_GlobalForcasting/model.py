from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tcn import TCN 

def create_model_tcn(lag: int, dense_neurons: int, look_forward: int) -> Model:
    """
    Creates a Temporal Convolutional Network (TCN) model for time series forecasting.

    Args:
        lag (int): Input sequence length (number of past time steps).
        dense_neurons (int): Number of neurons in the dense layer.
        look_forward (int): Forecasting horizon.

    Returns:
        Model: Compiled TCN model.
    """

    # Input Layer
    input_layer = layers.Input(shape=(1, lag), name="Input-Layer")

    # TCN Layer
    tcn_layer = TCN(return_sequences=True, dilations=[1, 2, 4, 8], name="TCN-Layer")(input_layer)

    # Convolutional Layers
    conv1 = Conv1D(filters=64, kernel_size=4, strides=2, activation="relu", padding="same", name="Conv1D-1")(tcn_layer)
    conv2 = Conv1D(filters=16, kernel_size=4, strides=2, activation="relu", padding="same", name="Conv1D-2")(conv1)

    # Flatten Layer
    flatten_layer = Flatten(name="Flatten-Layer")(conv2)

    # Fully Connected Layer
    dense_layer = Dense(units=dense_neurons, activation="relu", name="Dense-Layer")(flatten_layer)

    # Output Layer
    output_layer = Dense(units=look_forward, activation="linear", name="Output-Layer")(dense_layer)

    # Define Model
    model = Model(inputs=input_layer, outputs=output_layer, name="TCN_Model")

    return model