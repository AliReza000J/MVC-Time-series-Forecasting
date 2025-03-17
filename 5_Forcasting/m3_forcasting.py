import os
import csv
import numpy as np
from texttable import Texttable
from run_model import run_local_models

file_list=[]

def get_param2():
    """
    Returns dataset-specific parameters for forecasting experiments.

    Returns:
        dataset_name (str): Name of the dataset.
        num_clusters (int): Number of clusters for FCM (Fuzzy C-Means).
        batch_sizes (list): List of batch sizes for training.
        epochs (list): List of epoch values for training.
        model_name (str): Model identifier string.
        mase_freq (int): Frequency parameter for MASE calculation.
    """

    # Define number of clusters per dataset
    num_clusters_dict = {
        'm3-demo': 2, 'm3-finance': 2, 'm3-industry': 2,
        'm3-micro': 2, 'm3-macro': 2, 'm3-other': 2
    }

    # Define batch size and epochs per dataset
    training_params = {
        'm3-demo': {'batch': [40], 'epoch': [5]},
        'm3-micro': {'batch': [40], 'epoch': [5]},
        'm3-industry': {'batch': [40], 'epoch': [5]},
        'm3-macro': {'batch': [60], 'epoch': [10]},
        'm3-other': {'batch': [20], 'epoch': [20]},
        'm3-finance': {'batch': [20], 'epoch': [20]}
    }

    # Select dataset (change as needed)
    dataset_name = 'm3-finance'

    # Retrieve dataset-specific parameters
    num_clusters = num_clusters_dict.get(dataset_name, 1)  # Default to 1 if not found
    batch_sizes = training_params[dataset_name]['batch']
    epochs = training_params[dataset_name]['epoch']

    # Define MASE frequency (used for error metrics)
    mase_freq = 12

    # Define model naming convention
    model_name = f"{dataset_name}_fcm_raw_16_NR2_NO_FeWe"

    return dataset_name, num_clusters, batch_sizes, epochs, model_name, mase_freq


def m3(AEName="", Dim=0, run=1):
    """
    Run forecasting experiments on the M3 dataset with different model configurations.

    Parameters:
        AEName (str): Name of the AutoEncoder model (if used).
        Dim (int): Dimensionality of the extracted features.
        run (int): Identifier for multiple runs of experiments.

    Returns:
        predicted_values (list): Model's predictions.
        TestY (list): Ground truth values for the test set.
        TrainY (list): Ground truth values for the training set.
    """

    # Retrieve dataset name and model parameters
    dataset_name, num_cluster, batch_sizes, epochs, mdl, mase_freq = get_param2()

    # Construct CSV file name for results storage
    csv_file = f"{AEName}_{Dim}.csv"

    # Determine file mode (append if exists, otherwise write)
    file_mode = 'a' if os.path.exists(csv_file) else 'w'

    # Store the file in a global list (assuming file_list is defined elsewhere)
    file_list.append(csv_file)

    with open(csv_file, mode=file_mode, newline='') as file:
        writer = csv.writer(file)

        # Iterate through all combinations of epochs and batch sizes
        for epoch in epochs:
            for batch in batch_sizes:
                # Skip a specific configuration (500 epochs and 200 batch size)
                if epoch == 500 and batch == 200:
                    continue

                # Run the forecasting model
                results, initial_data, predicted_values, TestY, TrainY, test_predictions = run_local_models(
                    dataset_name=dataset_name, 
                    number_of_clusters=num_cluster, 
                    AEName=AEName, 
                    Dim=Dim, 
                    epochs=epoch, 
                    batch=batch, 
                    use_saved_model=False, 
                    save_trained_model=False, 
                    run=run
                )

                # Extract performance metrics
                performance_metrics = {
                    'val_SMAPE': results['val_SMAPE'],
                    'val_RMSE': results['val_RMSE'],
                    'val_MASE': results['val_MASE'],
                    'test_SMAPE': results['test_SMAPE'],
                    'test_RMSE': results['test_RMSE'],
                    'test_MASE': results['test_MASE']
                }

                # Display results in a formatted table
                table = Texttable()
                table.add_rows([
                    ['Dataset', 'Mean sMAPE', 'Median sMAPE', 'Mean RMSE', 'Median RMSE', 'Mean MASE', 'Median MASE'],
                    ['Validation', 
                     np.mean(performance_metrics['val_SMAPE']), np.median(performance_metrics['val_SMAPE']),
                     np.mean(performance_metrics['val_RMSE']), np.median(performance_metrics['val_RMSE']),
                     np.mean(performance_metrics['val_MASE']), np.median(performance_metrics['val_MASE'])
                    ],
                    ['Test', 
                     np.mean(performance_metrics['test_SMAPE']), np.median(performance_metrics['test_SMAPE']),
                     np.mean(performance_metrics['test_RMSE']), np.median(performance_metrics['test_RMSE']),
                     np.mean(performance_metrics['test_MASE']), np.median(performance_metrics['test_MASE'])
                    ]
                ])

                # Save initial data results to CSV
                writer.writerows(initial_data)

                # Print the results table
                print(table.draw())

    # Save test predictions to a CSV file dynamically in the working directory
    test_predictions.to_csv(f'./{dataset_name}-test-predictions-{run}.csv')

    return predicted_values, TestY, TrainY


def run_experiments(n_runs=2, ae_name="", dim=0):
    """
    Runs multiple experiments and collects predicted values across runs.

    Args:
        n_runs (int): Number of runs to perform.
        ae_name (str): Name of the autoencoder model.
        dim (int): Dimensionality reduction parameter.

    Returns:
        np.ndarray: Combined predictions from all runs.
    """
    predicted_values_all_runs = []  

    for run in range(n_runs):
        # Retrieve dataset parameters
        dataset_name, num_cluster, batch, epoch, model_name, mase_freq = get_param2()

        # Run the model and obtain predictions
        predicted_values, TestY, TrainY = m3(model_name, dim, run=run)

        # Store predictions
        predicted_values_all_runs.append(predicted_values)

    # Convert list to numpy array
    combined_predictions = np.stack(predicted_values_all_runs, axis=0)

    print("Shape of combined predictions:", combined_predictions.shape)
    
    return combined_predictions
