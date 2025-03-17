import os
import gc
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import pandas as pd
from texttable import Texttable
from tensorflow import keras
from metrics import smape, root_mean_squared_error, mase_val, mase
from preprocessing import preprocess_dataset, rescale_data_to_main_value, normalize_dataset, stl_decomposition
from model import create_model_tcn
from get_m3_dataset_params import get_m3_dataset_params

# Configure logging
logging.basicConfig(
    filename="model_training.log",  # Log file
    level=logging.INFO,  # Log level: INFO and above
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w"  # Overwrites the log file on each run
)

console_handler = logging.StreamHandler()  # Console output handler
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

def run_model_test(dataset, data_means, dataset_seasonal, dataset_name, cluster_label, 
                   lag, look_forward, sample_overlap, batch_size, epochs, learning_rate, 
                   suilin_smape, dataset_path, frequency, use_saved_model=False, save_trained_model=False):
    
    print("Dataset size:", len(dataset))

    look_back = lag
    calculation_method = 'per_series'  # Options: 'single_value' | 'per_series'

    # Data Preprocessing (Augmented)
    trainX, valX, testX, trainY, valY, testY, test_means, val_means, val_seasonal, test_seasonal, test_seasonal2 = \
        preprocess_dataset(dataset, lag, look_forward, sample_overlap, data_means, dataset_seasonal, frequency, dataset_name)

    model_path = f"{dataset_path}/{dataset_name}-model-cluster-{cluster_label}"

    # Check if we should use a pre-trained model
    if use_saved_model and os.path.exists(model_path):
        model = keras.models.load_model(model_path)
    else:
        use_saved_model = False  # No pre-trained model available
        save_trained_model = True  # Enable saving if training is required

    if use_saved_model:
        # Perform predictions using the saved model
        val_predictions = model.predict([valX], batch_size=16, verbose=0)
        test_predictions = model.predict([testX], batch_size=16, verbose=0)

        # Compute validation & test metrics
        val_RMSE = root_mean_squared_error(valY, val_predictions, calculation_method)
        val_SMAPE = smape(valY, val_predictions, calculation_method, suilin_smape)
        test_RMSE = root_mean_squared_error(testY, test_predictions, calculation_method)
        test_SMAPE = smape(testY, test_predictions, calculation_method, suilin_smape)

    else:
        # Train the model from scratch
        dense_neurons = 100
        activation_function = 'linear'
        output_activation = 'linear'
        
        print("---------------------------------------------------------------------")
        print(f"Lag: {lag}, Look Forward: {look_forward}, Sample Overlap: {sample_overlap}")
        print(f"Train shape: {trainX.shape}, Validation shape: {valX.shape}, Test shape: {testX.shape}")
        print(f"Learning Rate: {learning_rate}, Dense Neurons: {dense_neurons}, Activation: {activation_function}")

        validation_loss = []
        test_loss = []
        iterations = 1  # Number of iterations for model training
        
        for _ in range(iterations):
            model = create_model_tcn(lag, dense_neurons, look_forward)
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            
            # Compile model
            model.compile(loss="mse", optimizer=optimizer, metrics=["mse"])

            # Train model for specified epochs
            for _ in range(epochs):
                model.fit([trainX], trainY, validation_data=([valX, valY]), verbose=0, batch_size=batch_size)
                
                # Generate predictions
                val_predictions = model.predict([valX], batch_size=16, verbose=0)
                test_predictions = model.predict([testX], batch_size=16, verbose=0)

                # Compute evaluation metrics
                val_RMSE = root_mean_squared_error(valY, val_predictions, calculation_method)
                val_SMAPE = smape(valY, val_predictions, calculation_method, suilin_smape)
                test_RMSE = root_mean_squared_error(testY, test_predictions, calculation_method)
                test_SMAPE = smape(testY, test_predictions, calculation_method, suilin_smape)

                validation_loss.append(np.mean(val_RMSE))
                test_loss.append(np.mean(test_RMSE))

            # Clear Keras session and garbage collect
            K.clear_session()
            gc.collect()

            # Save model if enabled
            if save_trained_model:
                model.save(model_path)

    # Rescale predictions back to original scale
    rescaled_valY = rescale_data_to_main_value(valY, val_means, val_seasonal)
    rescaled_val_predictions = rescale_data_to_main_value(val_predictions, val_means, val_seasonal)
    
    rescaled_testY = rescale_data_to_main_value(testY, test_means, test_seasonal)
    rescaled_trainY = rescale_data_to_main_value(dataset, data_means, dataset_seasonal)

    # Compute final performance metrics
    val_SMAPE = smape(rescaled_valY, rescaled_val_predictions, calculation_method, suilin_smape)
    val_RMSE = root_mean_squared_error(rescaled_valY, rescaled_val_predictions, calculation_method)
    val_MASE = mase_val(rescaled_valY, rescaled_val_predictions, rescaled_trainY, frequency)

    rescaled_test_predictions = rescale_data_to_main_value(test_predictions, test_means, test_seasonal2)
    test_SMAPE = smape(rescaled_testY, rescaled_test_predictions, calculation_method)
    test_RMSE = root_mean_squared_error(rescaled_testY, rescaled_test_predictions, calculation_method)
    test_MASE = mase(rescaled_testY, rescaled_test_predictions, rescaled_trainY, frequency)

    # Print final prediction results
    print("Final Predictions:", rescaled_test_predictions.shape)

    # Store results in a structured format
    results = {
        'val_SMAPE': val_SMAPE,
        'val_RMSE': val_RMSE,
        'val_MASE': val_MASE,
        'test_SMAPE': test_SMAPE,
        'test_RMSE': test_RMSE,
        'test_MASE': test_MASE,
        'predicted': rescaled_test_predictions,
        'Train': rescaled_trainY,
        'Test': rescaled_testY
    }

    return results, rescaled_test_predictions, rescaled_testY

def run_local_models(dataset_name, number_of_clusters=2, AEName='LSTM', Dim=8, epochs=20, batch=20, 
                     use_saved_model=False, save_trained_model=False, run=1):
    """
    Runs local models for time-series forecasting using clustering.

    Args:
        dataset_name (str): Name of the dataset.
        number_of_clusters (int): Number of clusters for time-series segmentation.
        AEName (str): Type of Autoencoder used.
        Dim (int): Latent dimension of Autoencoder.
        epochs (int): Number of training epochs.
        batch (int): Batch size.
        use_saved_model (bool): Whether to load a pre-trained model.
        save_trained_model (bool): Whether to save the trained model.
        run (int): Experiment run identifier.

    Returns:
        results (dict): Validation and test evaluation metrics.
        initial_data (list): Summary of dataset and training parameters.
        test_predictions_list (list): List of predictions for test data.
        TestY (np.ndarray): Ground truth test values.
        TrainY (np.ndarray): Training data labels.
        test_predictions_df (pd.DataFrame): DataFrame of sorted test predictions.
    """

    # Garbage collection to free memory
    gc.collect()

    print('Dataset:', dataset_name)

    # Prepare & Read Dataset Parameters
    dataset, labels, lag, look_forward, sample_overlap, learning_rate, dataset_path, suilin_smape, frequency = get_m3_dataset_params(dataset_name)

    # Normalize dataset and extract mean values
    dataset, data_means = normalize_dataset(dataset, look_forward)

    # Apply Seasonal-Trend Decomposition (STL)
    dataset, seasonal, trend = stl_decomposition(dataset, frequency, look_forward)

    # Initialize variables
    predicted_Values, TestY, TrainY = np.array([]), np.array([]), np.array([])

    # Cluster Series Based on Feature Vectors
    clusters = np.zeros(len(labels)) if number_of_clusters == 1 else labels
    dataset = np.array(dataset)

    results = {
        'val_SMAPE': np.array([]),
        'val_RMSE': np.array([]),
        'val_MASE': np.array([]),
        'test_SMAPE': np.array([]),
        'test_RMSE': np.array([]),
        'test_MASE': np.array([])
    }

    # Store test predictions and true values for all clusters
    all_test_predictions = []
    all_y_test = []

    # Loop Through Clusters
    for cluster_label in range(number_of_clusters):
        # Extract data for the current cluster
        idx = [x for x in range(len(clusters)) if clusters[x] == cluster_label]
        cluster_dataset = dataset[idx]
        cluster_dataset_means = data_means[idx]
        cluster_dataset_seasonal = seasonal[idx]

        # Train and test model for the cluster
        result, test_predictions, y_test = run_model_test(
            cluster_dataset, cluster_dataset_means, cluster_dataset_seasonal, dataset_name, cluster_label, 
            lag, look_forward, sample_overlap, batch, epochs, learning_rate, suilin_smape, dataset_path, frequency, 
            use_saved_model, save_trained_model
        )

        # Store predictions for this cluster
        all_test_predictions.append((idx, test_predictions))  
        all_y_test.append((idx, y_test))

        # Merge results across clusters
        for key in results.keys():
            results[key] = np.concatenate((results[key], result[key]))

        # Store predictions for full dataset
        if cluster_label == 0:
            predicted_Values = result['predicted']
            TestY = result['Test']
            TrainY = result['Train']
        else:
            predicted_Values = np.concatenate((predicted_Values, result['predicted']))
            TestY = np.concatenate((TestY, result['Test']))
            TrainY = np.concatenate((TrainY, result['Train']))

    # Flatten all predictions into a sorted DataFrame
    flattened_test_predictions = [(idx[i], preds[i]) for idx, preds in all_test_predictions for i in range(len(preds))]
    flattened_y_test = [(idx[i], preds[i]) for idx, preds in all_y_test for i in range(len(preds))]

    test_predictions_df = pd.DataFrame(flattened_test_predictions, columns=['Original_Index', 'Prediction']).sort_values(by='Original_Index').reset_index(drop=True)
    y_test_df = pd.DataFrame(flattened_y_test, columns=['Original_Index', 'values']).sort_values(by='Original_Index').reset_index(drop=True)

    test_predictions_list = test_predictions_df['Prediction'].tolist()  # Convert predictions to list

    # Debugging Output
    print("Prediction Shape:", predicted_Values.shape)

    # Save Results to Disk
    metric_names = ['SMAPE', 'RMSE', 'MASE']
    for metric in metric_names:
        np.save(f"val_{metric}_{run}_{dataset_name}_{batch}_{epochs}.npy", results[f'val_{metric}'])
        np.save(f"test_{metric}_{run}_{dataset_name}_{batch}_{epochs}.npy", results[f'test_{metric}'])

    # Display Results in Table Format
    print('\n\n#------------------------------------ Scaled Results ------------------------------------#')
    t = Texttable()
    t.add_rows([
        ['Index', 'Mean sMAPE', 'Median sMAPE', 'Mean RMSE', 'Median RMSE', 'Mean MASE'],
        ['Validate', np.mean(results['val_SMAPE']), np.median(results['val_SMAPE']), np.mean(results['val_RMSE']), 
                     np.median(results['val_RMSE']), np.mean(results['val_MASE'])],
        ['Test', np.mean(results['test_SMAPE']), np.median(results['test_SMAPE']), np.mean(results['test_RMSE']), 
                 np.median(results['test_RMSE']), np.mean(results['test_MASE'])]
    ])
    print(t.draw())

    # Display Training Configuration
    t = Texttable()
    t.add_rows([
        ['Dataset Name', 'Run', 'N. Epochs', 'Batch Size', 'Auto Encoder', 'Latent Dimension'],
        [dataset_name, run, epochs, batch, AEName, Dim]
    ])
    print(t.draw())

    # Generate Initial Data Summary for CSV Export
    initial_data = [
        ['Dataset Name', 'Run', 'N. Epochs', 'Batch Size', 'Auto Encoder', 'Latent Dimension', 'Index', 'Mean sMAPE', 
         'Median sMAPE', 'Mean RMSE', 'Median RMSE', 'Mean MASE', 'Median MASE'],
        [dataset_name, run, epochs, batch, AEName, Dim, 'Validate', np.mean(results['val_SMAPE']), 
         np.median(results['val_SMAPE']), np.mean(results['val_RMSE']), np.median(results['val_RMSE']), 
         np.mean(results['val_MASE']), np.median(results['val_MASE'])],
        [dataset_name, run, epochs, batch, AEName, Dim, 'Test', np.mean(results['test_SMAPE']), np.median(results['test_SMAPE']), 
         np.mean(results['test_RMSE']), np.median(results['test_RMSE']), np.mean(results['test_MASE']), np.median(results['test_MASE'])]
    ]

    # Save Test Labels to CSV
    y_test_df.to_csv('/kaggle/working/y_test.csv', index=False)

    return results, initial_data, test_predictions_list, TestY, TrainY, test_predictions_df