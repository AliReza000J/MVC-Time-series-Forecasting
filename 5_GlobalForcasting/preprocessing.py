import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL

def normalize_dataset(dataset, look_forward):
    """Applies mean normalization and log transformation to a dataset."""
    data_means = []

    for index in range(len(dataset)):
        series = dataset[index]

        # Mean of training data (excluding the last 'look_forward' points)
        series_mean = np.mean(series[:-look_forward]) if len(series) > look_forward else np.mean(series)
        series_mean = max(series_mean, 0.001)  # Avoid zero division

        data_means.append(series_mean)
        dataset[index] = series / series_mean

        # Log Transformation
        dataset[index] = np.log1p(np.clip(dataset[index], 0, None))

    return dataset, np.array(data_means)

def rescale_data_to_main_value(data, means, dataset_seasonal=[]):
    """Reverts log transformation and mean normalization."""
    for index in range(len(data)):
        if dataset_seasonal:
            data[index] += dataset_seasonal[index]  # Add back seasonal component

        data[index] = np.expm1(data[index])  # Inverse of log1p
        data[index] *= means[index]  # Revert normalization

    return data

def normalize_feature_vectors(features):
    """Applies Min-Max Normalization to feature vectors."""
    minimum = features.min(axis=0)
    maximum = features.max(axis=0)

    features = (features - minimum) / (maximum - minimum + 1e-8)  # Avoid division by zero
    return features

# Dataset Creation for Time-Series Forecasting
def create_dataset(sample, look_back, look_forward, sample_overlap, dataset_seasonal):
    """Creates sliding window samples for training."""
    if sample_overlap >= look_forward or sample_overlap < 0:
        sample_overlap = look_forward - 1
    if look_forward == 1:
        sample_overlap = 0

    dataX, dataY, dataY_seasonal = [], [], []
    
    for i in range(0, len(sample) - look_back - look_forward + 1, look_forward - sample_overlap):
        dataX.append(sample[i:(i + look_back), 0])
        dataY.append(sample[(i + look_back):(i + look_back + look_forward), 0])
        dataY_seasonal.append(dataset_seasonal[(i + look_back):(i + look_back + look_forward)])

    return np.array(dataX), np.array(dataY), np.array(dataY_seasonal)

def create_dataset2(sample, look_back, look_forward, sample_overlap, dataset_seasonal, dataset_name):
    """Creates dataset with trend & seasonal forecasting using Exponential Smoothing."""
    if sample_overlap >= look_forward or sample_overlap < 0:
        sample_overlap = look_forward - 1
    if look_forward == 1:
        sample_overlap = 0

    sample_trn = sample[:-look_forward]
    frequency = {'tourism': 4, 'cif-6': 12}.get(dataset_name, 12)

    # Handle cases with shorter sequences
    if len(sample_trn) > 2 * frequency:
        model = ExponentialSmoothing(sample_trn.flatten(), seasonal_periods=frequency, trend='add', seasonal='add').fit()
    elif len(sample_trn) > frequency:
        model = ExponentialSmoothing(sample_trn.flatten(), seasonal_periods=frequency // 2, trend='add', seasonal='add').fit()
    else:
        model = ExponentialSmoothing(sample_trn.flatten()).fit()

    forecast = model.forecast(steps=look_forward).reshape(1, -1)

    # Combine training & forecasted values
    augmented_series = np.concatenate([sample_trn[-(look_back + look_forward - 1):].reshape(-1, 1), forecast.T])
    augmented_series = augmented_series.flatten()

    aug_trainX, aug_trainY = [], []

    for i in range(0, len(augmented_series) - look_back - look_forward + 1, look_forward - sample_overlap):
        aug_trainX.append(augmented_series[i:(i + look_back)])
        aug_trainY.append(augmented_series[(i + look_back):(i + look_back + look_forward)])

    return np.array(aug_trainX), np.array(aug_trainY)

def stl_decomposition(dataset: np.ndarray, frequency: int, look: int):
    """
    Decomposes a dataset using STL (Seasonal-Trend decomposition) and removes the seasonal component.
    
    Args:
        dataset (np.ndarray): Input time series dataset (2D array).
        frequency (int): Seasonal frequency of the time series.
        look (int): Number of future time steps to adjust.
    
    Returns:
        tuple: Adjusted dataset, seasonal components, trend components.
    """
    seasonal, trend = [], []
    
    for index in range(len(dataset)):
        if frequency is not None:
            stl = STL(dataset[index][:-look], period=frequency, seasonal="periodic").fit()
            stl_full = STL(dataset[index], period=frequency, seasonal="periodic").fit()

            seasonal.append(np.concatenate([stl.seasonal, stl_full.seasonal[-look:]]))
            trend.append(np.concatenate([stl.trend, stl_full.trend[-look:]]))
            
            dataset[index] -= np.concatenate([stl.seasonal, stl_full.seasonal[-look:]])
        else:
            seasonal.append(np.zeros_like(dataset[index]))
            trend.append(np.zeros_like(dataset[index]))

    return dataset, np.array(seasonal), np.array(trend)

def create_sample(
    look_forward: int,
    sample_seasonal: np.ndarray,
    dataX: np.ndarray, 
    dataY: np.ndarray, 
    data_mean: float, 
    dataY_seasonal: np.ndarray, 
    frequency: int
):
    """
    Creates training, validation, and test sets for time series forecasting.

    Args:
        look_forward (int): Forecasting horizon.
        sample_seasonal (np.ndarray): Seasonal component of the dataset.
        dataX (np.ndarray): Input features.
        dataY (np.ndarray): Target values.
        data_mean (float): Mean of the dataset.
        dataY_seasonal (np.ndarray): Seasonal values of target.
        frequency (int): Seasonal period.

    Returns:
        tuple: Train, validation, and test datasets along with seasonal components.
    """
    test_size, val_size = 1, 1
    train_size = len(dataX) - test_size
    train_size0 = train_size - look_forward + 1

    trainX, testX = dataX[:train_size0-val_size], dataX[train_size:]
    trainY, testY = dataY[:train_size0-val_size], dataY[train_size:]

    valX, valY = dataX[train_size0-val_size:train_size0], dataY[train_size0-val_size:train_size0]

    trainX, valX, testX = map(lambda x: x.reshape(x.shape[0], 1, x.shape[1]), [trainX, valX, testX])

    val_means = np.full(len(valY), data_mean)
    test_means = np.full(len(testY), data_mean)
    
    val_seasonal = dataY_seasonal[train_size0-val_size:train_size0]
    
    sample_size = len(sample_seasonal.flatten()) - look_forward
    train3 = sample_seasonal[:sample_size].flatten()

    if frequency:
        sp = frequency if len(train3) > 2 * frequency else max(1, frequency // 2)
        fit = ExponentialSmoothing(pd.Series(train3), seasonal_periods=sp, trend='add', seasonal='add').fit()
        preds2 = fit.forecast(steps=look_forward).values.reshape(1, -1)
    else:
        preds2 = np.zeros((1, look_forward))

    test_seasonal_y = dataY_seasonal[train_size:]

    return (trainX, valX, testX, trainY, valY, testY, 
            test_means, val_means, val_seasonal, test_seasonal_y, preds2)

def preprocess_dataset(
    all_dataset: list, 
    lag: int, 
    look_forward: int, 
    sample_overlap: int, 
    data_means: list, 
    dataset_seasonal: list,
    frequency: int
):
    """
    Prepares the dataset for training, validation, and testing by processing multiple time series.

    Args:
        all_dataset (list): List of time series datasets.
        lag (int): Window size for input sequences.
        look_forward (int): Forecasting horizon.
        sample_overlap (int): Overlapping samples.
        data_means (list): Mean values for each dataset.
        dataset_seasonal (list): Seasonal components for each dataset.
        frequency (int): Seasonal frequency.

    Returns:
        tuple: Processed train, validation, and test datasets.
    """
    trainX, trainY, valX, valY, testX, testY = [], [], [], [], [], []
    all_test_means, all_val_means, all_test_seasonals, all_test_seasonals2, all_val_seasonals = [], [], [], [], []

    for index, series in enumerate(all_dataset):
        sample = np.array(series).reshape(-1, 1)
        dataX_s, dataY_s, dataY_seasonal = create_dataset(sample, lag, look_forward, sample_overlap, dataset_seasonal[index])

        temp_data = create_sample(look_forward, dataset_seasonal[index], dataX_s, dataY_s, data_means[index], dataY_seasonal, frequency)

        temp_trainX, temp_valX, temp_testX, temp_trainY, temp_valY, temp_testY, test_means, val_means, val_seasonal, test_seasonal, preds2 = temp_data
        
        trainX.extend(temp_trainX.tolist())
        trainY.extend(temp_trainY.tolist())
        valX.extend(temp_valX.tolist())
        valY.extend(temp_valY.tolist())
        testX.extend(temp_testX.tolist())
        testY.extend(temp_testY.tolist())

        all_test_means.extend(test_means.tolist())
        all_val_means.extend(val_means.tolist())
        all_test_seasonals.extend(test_seasonal.tolist())
        all_test_seasonals2.extend(preds2.tolist())
        all_val_seasonals.extend(val_seasonal.tolist())

    return (np.array(trainX), np.array(valX), np.array(testX),
            np.array(trainY), np.array(valY), np.array(testY),
            np.array(all_test_means), np.array(all_val_means),
            np.array(all_val_seasonals), np.array(all_test_seasonals), 
            np.array(all_test_seasonals2))

def save_prediction_result(data: np.ndarray, dataset_name: str = 'dataset', dataset_path: str = ''):
    """
    Saves the prediction results to a CSV file.

    Args:
        data (np.ndarray): Predicted values.
        dataset_name (str): Name of the dataset.
        dataset_path (str): Path to save the CSV file.
    """
    filename = f"{dataset_path}/{dataset_name}-results.csv" if dataset_name else "results.csv"
    pd.DataFrame(data).to_csv(filename, index=False, header=False)
