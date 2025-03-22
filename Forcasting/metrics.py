import numpy as np

# Root Mean Squared Error (RMSE) Calculation
def root_mean_squared_error(actual, forecast, method='single_value'):
    """
    Computes RMSE for time-series predictions.

    Args:
        actual (np.ndarray): Ground truth values.
        forecast (np.ndarray): Predicted values.
        method (str): 'single_value' computes a single RMSE value,
                      otherwise returns RMSE for each time series separately.

    Returns:
        float or list: RMSE value(s).
    """
    if method == 'single_value':
        return np.sqrt(np.mean((actual.flatten() - forecast.flatten())**2))
    
    return [np.sqrt(np.mean((actual[i] - forecast[i])**2)) for i in range(len(actual))]


# Symmetric Mean Absolute Percentage Error (SMAPE) Calculation
def single_point_smape(actual, forecast, suilin_smape=False):
    """
    Calculates SMAPE for a single prediction point.

    Args:
        actual (float or np.ndarray): Actual values.
        forecast (float or np.ndarray): Forecasted values.
        suilin_smape (bool): Whether to use an adjusted denominator.

    Returns:
        float: SMAPE value.
    """
    epsilon = 0.1 if suilin_smape else 0  # Adjustment to prevent division by zero
    denominator = np.abs(actual) + np.abs(forecast) + epsilon
    return np.sum(2 * np.abs(forecast - actual) / np.maximum(denominator, 0.5 + epsilon))


def smape(actual, forecast, method='single_value', suilin_smape=False):
    """
    Computes SMAPE for time-series predictions.

    Args:
        actual (np.ndarray): Ground truth values.
        forecast (np.ndarray): Predicted values.
        method (str): 'single_value' computes a single SMAPE value,
                      otherwise returns SMAPE for each time series separately.
        suilin_smape (bool): Whether to use an adjusted denominator.

    Returns:
        float or np.ndarray: SMAPE value(s).
    """
    if method == 'single_value':
        return 100 * np.mean([single_point_smape(actual[i], forecast[i], suilin_smape) for i in range(len(actual))])
    
    return np.array([100 * np.mean([single_point_smape(actual[i, j], forecast[i, j], suilin_smape)
                                    for j in range(len(actual[i]))]) for i in range(len(actual))])


# Mean Absolute Scaled Error (MASE) Calculation
def single_point_mase(actual, forecast, insample, frequency=12):
    """
    Computes MASE for a single prediction point.

    Args:
        actual (float or np.ndarray): Ground truth values.
        forecast (float or np.ndarray): Predicted values.
        insample (np.ndarray): Training data for scaling.
        frequency (int): Seasonal frequency (default: 12 for monthly data).

    Returns:
        float: MASE value.
    """
    return np.mean(np.abs(forecast - actual)) / np.mean(np.abs(insample[:-frequency] - insample[frequency:]))


def mase(actual, forecast, insample, frequency=12):
    """
    Computes MASE for time-series predictions.

    Args:
        actual (np.ndarray): Ground truth values.
        forecast (np.ndarray): Predicted values.
        insample (np.ndarray): Training data for scaling.
        frequency (int): Seasonal frequency (default: 12 for monthly data).

    Returns:
        np.ndarray: MASE values for each time series.
    """
    print("Shapes:", actual.shape, forecast.shape, insample.shape)

    mase_values = []
    for i in range(len(actual)):
        sum_mase = 0
        for j in range(len(actual[i])):
            sum_mase += single_point_mase(actual[i, j], forecast[i, j], insample[i][:-len(actual[i])], frequency)
        mase_values.append(sum_mase / len(actual[i]))

    return np.array(mase_values)


def mase_val(actual, forecast, insample, frequency=12):
    """
    Computes MASE for validation data, considering a longer insample period.

    Args:
        actual (np.ndarray): Ground truth values.
        forecast (np.ndarray): Predicted values.
        insample (np.ndarray): Training data for scaling.
        frequency (int): Seasonal frequency (default: 12 for monthly data).

    Returns:
        np.ndarray: MASE values for each validation set.
    """
    print("Shapes:", actual.shape, forecast.shape, insample.shape)

    mase_values = []
    for i in range(len(actual)):
        sum_mase = 0
        for j in range(len(actual[i])):
            sum_mase += single_point_mase(actual[i, j], forecast[i, j], insample[i][:-2 * len(actual[i])], frequency)
        mase_values.append(sum_mase / len(actual[i]))

    return np.array(mase_values)