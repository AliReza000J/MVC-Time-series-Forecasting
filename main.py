import numpy as np
import pandas as pd
from ViewGeneration.view_generation import process_m3_dataset  # M3 dataset preprocessing
from ViewGeneration.view_generation import process_time_series  # Generate views

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
    data = process_m3_dataset(file_path)

    # Step 2: Generate images from time series

    #  micro
    micro = data['MICRO']
    print(micro)
    process_time_series(micro, "/workspaces/MVC-Time-series-Forcasting/data/generated_images/MICRO")

    #  industry
    industry = data['INDUSTRY']
    print(industry)
    process_time_series(industry, "/workspaces/MVC-Time-series-Forcasting/data/generated_images/INDUSTRY/", 18, True, 474)

    #  macro
    macro = data['MACRO']
    print(macro)
    process_time_series(macro, "/workspaces/MVC-Time-series-Forcasting/data/generated_images/MACRO", 18, True, 808)

    #  finance
    finance = data['FINANCE']
    print(finance)
    process_time_series(finance, "/workspaces/MVC-Time-series-Forcasting/data/generated_images/FINANCE", 18, True, 1120)

     #  demo
    demo = data['DEMOGRAPHIC']
    print(demo)
    process_time_series(demo, "/workspaces/MVC-Time-series-Forcasting/data/generated_images/DEMOGRAPHIC", 18, True, 1265)

     #  other
    other = data['OTHER']
    print(other)
    process_time_series(other, "/workspaces/MVC-Time-series-Forcasting/data/generated_images/OTHER", 18, True, 1376)

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