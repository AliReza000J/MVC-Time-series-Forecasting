import os
import numpy as np
import pandas as pd
from metrics import *

def median_ensemble_evaluation(combined_predictions, TestY, TrainY, dataset_name, mase_freq, save_path="/kaggle/working/ensemble_median"):
    """
    Performs median ensemble approach on predictions and evaluates performance.

    Args:
        combined_predictions (np.ndarray): Predictions from multiple runs.
        TestY (np.ndarray): Ground truth test values.
        TrainY (np.ndarray): Training values (for MASE calculation).
        dataset_name (str): Name of the dataset.
        mase_freq (int): Frequency parameter for MASE calculation.
        save_path (str): Directory to save results.

    Returns:
        dict: Final ensemble results (Mean & Median of SMAPE, RMSE, MASE).
    """
    os.makedirs(save_path, exist_ok=True)

    # Compute Median across runs
    median_predictions = np.median(combined_predictions, axis=0, keepdims=True)
    median_predictions_squeezed = np.squeeze(median_predictions, axis=0)

    # Define the calculation method
    calculations_method = 'per_series'

    # Compute Evaluation Metrics
    test_MASE = mase(TestY, median_predictions_squeezed, TrainY, mase_freq)
    test_SMAPE = smape(TestY, median_predictions_squeezed, calculations_method)
    test_RMSE = root_mean_squared_error(TestY, median_predictions_squeezed, calculations_method)

    # Print Results
    print("\nFor Median Ensemble Approach:")
    print("Mean SMAPE of ensemble:", np.mean(test_SMAPE))
    print("Mean RMSE of ensemble:", np.mean(test_RMSE))
    print("Mean MASE of ensemble:", np.mean(test_MASE))
    print("Median SMAPE of ensemble:", np.median(test_SMAPE))
    print("Median RMSE of ensemble:", np.median(test_RMSE))
    print("Median MASE of ensemble:", np.median(test_MASE))

    # Save Predictions & Metrics
    np.save(os.path.join(save_path, f"{dataset_name}_prediction_values_by_pools.npy"), combined_predictions)
    np.save(os.path.join(save_path, f"{dataset_name}_Median_prediction_values_by_ensemble.npy"), median_predictions_squeezed)
    np.save(os.path.join(save_path, f"{dataset_name}_SMAPE_of_ensemble.npy"), test_SMAPE)
    np.save(os.path.join(save_path, f"{dataset_name}_RMSE_of_ensemble.npy"), test_RMSE)
    np.save(os.path.join(save_path, f"{dataset_name}_MASE_of_ensemble.npy"), test_MASE)

    # Create Result Dictionary
    final_result = {
        'Mean_SMAPE_of_ensemble': np.mean(test_SMAPE),
        'Median_SMAPE_of_ensemble': np.median(test_SMAPE),
        'Mean_RMSE_of_ensemble': np.mean(test_RMSE),
        'Median_RMSE_of_ensemble': np.median(test_RMSE),
        'Mean_MASE_of_ensemble': np.mean(test_MASE),
        'Median_MASE_of_ensemble': np.median(test_MASE)
    }

    # Save Results to CSV
    result_df = pd.DataFrame(final_result, index=[0])
    result_df.to_csv(os.path.join(save_path, f"{dataset_name}_Ensemble_Result_median.csv"), index=False)

    return final_result


def mean_ensemble_evaluation(combined_predictions, TestY, TrainY, dataset_name, mase_freq, save_path="/kaggle/working/ensemble_mean"):
    """
    Performs mean ensemble approach on predictions and evaluates performance.

    Args:
        combined_predictions (np.ndarray): Predictions from multiple runs.
        TestY (np.ndarray): Ground truth test values.
        TrainY (np.ndarray): Training values (for MASE calculation).
        dataset_name (str): Name of the dataset.
        mase_freq (int): Frequency parameter for MASE calculation.
        save_path (str): Directory to save results.

    Returns:
        dict: Final ensemble results (Mean & Median of SMAPE, RMSE, MASE).
    """
    os.makedirs(save_path, exist_ok=True)

    # Compute Mean across runs
    mean_predictions = np.mean(combined_predictions, axis=0, keepdims=True)
    mean_predictions_squeezed = np.squeeze(mean_predictions, axis=0)

    # Define the calculation method
    calculations_method = 'per_series'

    # Compute Evaluation Metrics
    test_MASE = mase(TestY, mean_predictions_squeezed, TrainY, mase_freq)
    test_SMAPE = smape(TestY, mean_predictions_squeezed, calculations_method)
    test_RMSE = root_mean_squared_error(TestY, mean_predictions_squeezed, calculations_method)

    # Print Results
    print("\nFor Mean Ensemble Approach:")
    print("Mean SMAPE of ensemble:", np.mean(test_SMAPE))
    print("Mean RMSE of ensemble:", np.mean(test_RMSE))
    print("Mean MASE of ensemble:", np.mean(test_MASE))
    print("Median SMAPE of ensemble:", np.median(test_SMAPE))
    print("Median RMSE of ensemble:", np.median(test_RMSE))
    print("Median MASE of ensemble:", np.median(test_MASE))

    # Save Predictions & Metrics
    np.save(os.path.join(save_path, f"{dataset_name}_prediction_values_by_pools.npy"), combined_predictions)
    np.save(os.path.join(save_path, f"{dataset_name}_mean_prediction_values_by_ensemble.npy"), mean_predictions_squeezed)
    np.save(os.path.join(save_path, f"{dataset_name}_SMAPE_of_ensemble.npy"), test_SMAPE)
    np.save(os.path.join(save_path, f"{dataset_name}_RMSE_of_ensemble.npy"), test_RMSE)
    np.save(os.path.join(save_path, f"{dataset_name}_MASE_of_ensemble.npy"), test_MASE)

    # Create Result Dictionary
    final_result = {
        'Mean_SMAPE_of_ensemble': np.mean(test_SMAPE),
        'Median_SMAPE_of_ensemble': np.median(test_SMAPE),
        'Mean_RMSE_of_ensemble': np.mean(test_RMSE),
        'Median_RMSE_of_ensemble': np.median(test_RMSE),
        'Mean_MASE_of_ensemble': np.mean(test_MASE),
        'Median_MASE_of_ensemble': np.median(test_MASE)
    }

    # Save Results to CSV
    result_df = pd.DataFrame(final_result, index=[0])
    result_df.to_csv(os.path.join(save_path, f"{dataset_name}_Ensemble_Result_mean.csv"), index=False)

    return final_result