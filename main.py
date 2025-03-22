import numpy as np
import pandas as pd
from ViewGeneration.view_generation import process_time_series  # Generate views
from ViewGeneration.m3_preprocessing import process_m3_dataset  # M3 dataset preprocessing
# from feature_extraction import extract_time_series_features, extract_image_features  # Feature extraction
# from view_selection import select_views  # Select views
# from demo import fuzzy_multi_view_clustering  # Multi-view fuzzy clustering
# from m3_forcasting import run_experiments  # Run multiple experiments
# from ensemble import median_ensemble_evaluation  # Median ensemble

# ================================
#  Main Execution Script
# ================================
if __name__ == "__main__":
    """
    Unified pipeline:
    1. Load & process M3 dataset
    2. Generate images (RP, GAF, MTF) from time series
    3. Extract numerical features from time series
    4. Extract CNN features from images
    5. Select best views using diversity & similarity
    6. Perform multi-view fuzzy clustering
    7. Run multiple experiments & collect predictions
    8. Evaluate ensemble predictions using Median approach
    """

    # Step 1: Process M3 dataset
    file_path = "/workspaces/MVC-Time-series-Forcasting/data/M3Month.csv"
    processed_data = process_m3_dataset(file_path)

    if processed_data:
        print("Time-series processing complete. Categories:", list(processed_data.keys()))

        # Step 2: Generate images from time series
        image_output_path = "/workspaces/MVC-Time-series-Forcasting/data/generated_images/"
        for label, ts_data in processed_data.items():
            print(f"Generating images for category: {label}")
            # df = pd.DataFrame(ts_data)  # Convert to DataFrame for processing
            df = pd.DataFrame(ts_data.squeeze(axis=1))  # Convert to DataFrame for processing
            process_time_series(df, output_path=f"{image_output_path}/{label}")

    #     # Step 3: Extract numerical features from time series =!=!=!=
    #     ts_features = {label: extract_time_series_features(data) for label, data in processed_data.items()}
    #     np.save("time_series_features.npy", ts_features)
    #     print("Time-series feature extraction complete. Features saved.")

    #     # Step 4: Extract image-based CNN features
    #     img_features = {label: extract_image_features(f"{image_output_path}/{label}", gray=True) for label in processed_data.keys()}
    #     np.save("image_features.npy", img_features)
    #     print("Image feature extraction complete. Features saved.")

    #     # Step 5: Select best views
    #     selected_views = {}
    #     for label in processed_data.keys():
    #         feature_list = [ts_features[label], img_features[label]]
    #         selected_views[label] = select_views(feature_list, threshold=0.1)

    #     np.save("selected_views.npy", selected_views)
    #     print("View selection complete. Selected views saved.")

    #     # Step 6: Perform multi-view fuzzy clustering
    #     print("Running multi-view fuzzy clustering...")
    #     clustering_results = fuzzy_multi_view_clustering(dataset="demo")
    #     np.save("clustering_results.npy", clustering_results)
    #     print("Clustering completed. Results saved.")

    #     # Step 7: Run multiple experiments and collect predictions
    #     print("Running experiments...")
    #     combined_predictions, TestY, TrainY = run_experiments(n_runs=2, ae_name="Autoencoder_Model", dim=16)
    #     np.save("combined_predictions.npy", combined_predictions)
    #     print("Experiments completed. Predictions saved.")

    #     # Step 8: Perform Median Ensemble Evaluation
    #     print("Performing Median Ensemble Evaluation...")
    #     dataset_name = "M3_Monthly"
    #     mase_freq = 12  # Monthly data

    #     ensemble_results = median_ensemble_evaluation(
    #         combined_predictions=combined_predictions,
    #         TestY=TestY,
    #         TrainY=TrainY,
    #         dataset_name=dataset_name,
    #         mase_freq=mase_freq,
    #         save_path="ensemble_results"
    #     )

    #     print("Ensemble evaluation completed. Results saved.")
    #     print("Final Ensemble Results:", ensemble_results)

    # else:
    #     print("Error in processing M3 dataset. Please check file path and dataset format.")
