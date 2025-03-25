import os
import pandas as pd
import warnings
from Forecasting.m3_forecasting import get_param2

warnings.simplefilter(action='ignore', category=Warning)


def process_csv_results(directory_path='/kaggle/working/', dataset_name='m3-finance'):
    """
    Processes the last CSV file in the directory and calculates mean/median validation & test metrics.

    Args:
        directory_path (str): Path to the directory containing CSV files.
        dataset_name (str): Dataset name to retrieve parameters.

    Returns:
        None (Saves processed results as a new CSV file)
    """

    # List all files in the directory
    file_list = sorted([f for f in os.listdir(directory_path) if f.endswith('.csv')])

    if not file_list:
        print("No CSV files found in the directory.")
        return

    # Select the last CSV file
    csv_file = file_list[-1]
    file_path = os.path.join(directory_path, csv_file)
    print(f"Processing file: {csv_file}")

    # Read CSV
    try:
        df = pd.read_csv(file_path, header=None)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Retrieve dataset parameters
    dataset_name, num_cluster, batch_sizes, epochs, mdl, mase_freq = get_param2()
    
    results = []

    # Columns to process
    metric_cols = [7, 8, 9, 10, 11, 12]

    for epoch in epochs:
        for batch in batch_sizes:
            if epoch == 500 and batch == 200:
                continue

            # Filter validation and test data
            validate_df = df[(df[2] == str(epoch)) & (df[3] == str(batch)) & (df[6] == 'Validate')]
            test_df = df[(df[2] == str(epoch)) & (df[3] == str(batch)) & (df[6] == 'Test')]

            # Convert necessary columns to numeric, ignoring errors
            validate_df[metric_cols] = validate_df[metric_cols].apply(pd.to_numeric, errors='coerce')
            test_df[metric_cols] = test_df[metric_cols].apply(pd.to_numeric, errors='coerce')

            # Calculate mean values
            validate_mean = validate_df[metric_cols].mean()
            test_mean = test_df[metric_cols].mean()

            # Append results
            results.append({
                'Epoch': epoch,
                'Batch': batch,
                'Validate_Mean_sMAPE': validate_mean[7],
                'Validate_Median_sMAPE': validate_mean[8],
                'Validate_Mean_RMSE': validate_mean[9],
                'Validate_Median_RMSE': validate_mean[10],
                'Validate_Mean_MASE': validate_mean[11],
                'Validate_Median_MASE': validate_mean[12],
                'Test_Mean_sMAPE': test_mean[7],
                'Test_Median_sMAPE': test_mean[8],
                'Test_Mean_RMSE': test_mean[9],
                'Test_Median_RMSE': test_mean[10],
                'Test_Mean_MASE': test_mean[11],
                'Test_Median_MASE': test_mean[12]
            })

    # Convert results to DataFrame
    result_df = pd.DataFrame(results)

    # Sort by validation sMAPE
    result_df = result_df.sort_values(by='Validate_Mean_sMAPE', ascending=True)

    # Save results
    output_file = os.path.join(directory_path, f"Average_Results_{csv_file}")
    result_df.to_csv(output_file, index=False)

    print(f"Processed results saved to: {output_file}")