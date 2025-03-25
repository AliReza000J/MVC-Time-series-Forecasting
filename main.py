import numpy as np
from ViewGeneration.view_generation import process_m3_dataset  # M3 dataset preprocessing
from ViewGeneration.view_generation import process_time_series  # Generate views
from ViewEncoding.m3_preprocessing import process_m3_dataset # M3 dataset preprocessing for encoding
from ViewEncoding.feature_extraction import extract_time_series_features, extract_image_features  # Feature extraction
from sklearn.preprocessing import MinMaxScaler
from ViewSelection.view_selection import select_views  # View selection
from FuzzyClustering.demo import fuzzy_multi_view_clustering  # Multi-view clustering
from Forcasting.m3_forcasting import run_experiments  # Run forecasting experiments
from Forcasting.ensemble import median_ensemble_evaluation  # Median ensemble evaluation

min_max_scaler =MinMaxScaler()

if __name__ == "__main__":
    """
    Unified pipeline:
    1. Load & process M3 dataset
    2. Generate images (RP, GAF, MTF) from time series
    3.1. Extract numerical features from time series
    3.2. Extract CNN features from images
    4. Select best views using diversity & similarity
    5. Perform multi-view fuzzy clustering
    6. Run multiple experiments & collect predictions
    7. Evaluate ensemble predictions using Median approach
    """

    # Step 1: Process M3 dataset
    file_path = "/workspaces/MVC-Time-series-Forcasting/data/M3Month.csv"
    data = process_m3_dataset(file_path)

    # Step 2: Generate images from time series

    # #  micro
    # micro = data['MICRO']
    # print(micro)
    # process_time_series(micro, "/workspaces/MVC-Time-series-Forcasting/data/generated_images/MICRO")

    # #  industry
    # industry = data['INDUSTRY']
    # print(industry)
    # process_time_series(industry, "/workspaces/MVC-Time-series-Forcasting/data/generated_images/INDUSTRY", 18, True, 474)

    # #  macro
    # macro = data['MACRO']
    # print(macro)
    # process_time_series(macro, "/workspaces/MVC-Time-series-Forcasting/data/generated_images/MACRO", 18, True, 808)

    # #  finance
    # finance = data['FINANCE']
    # print(finance)
    # process_time_series(finance, "/workspaces/MVC-Time-series-Forcasting/data/generated_images/FINANCE", 18, True, 1120)

     #  demo
    demo = data['DEMOGRAPHIC']
    image_output_path = "/workspaces/MVC-Time-series-Forcasting/data/generated_images"
    process_time_series(demo, f'{image_output_path}/DEMOGRAPHIC', 18, True, 1265)

    #  #  other
    # other = data['OTHER']
    # print(other)
    # process_time_series(other, "/workspaces/MVC-Time-series-Forcasting/data/generated_images/OTHER", 18, True, 1376)

    # Step 3: Extract features using Autoencoder
    processed_data = process_m3_dataset(file_path)
    features_list = []
    if processed_data:
        print("Time-series processing complete. Categories:", list(processed_data.keys()))

    # Step 3.1: Extract numerical features only for 'DEMOGRAPHIC'
    demo_ts_features = {
        label: extract_time_series_features(data)
        for label, data in processed_data.items()
        if label == "DEMOGRAPHIC"
    }
    features_list.append(min_max_scaler.fit_transform(demo_ts_features))
    # np.save("demo_time_series_features.npy", demo_ts_features)
    print("Time-series feature extraction complete.")

    # Step 3.2: Extract image-based CNN features only for 'DEMOGRAPHIC'
    
    demo_rp_img_features = {
        label: extract_image_features(f"{image_output_path}/{label}/rp", gray=True)
        for label in processed_data.keys()
        if label == "DEMOGRAPHIC"
    }
    features_list.append(min_max_scaler.fit_transform(demo_rp_img_features))

    demo_mtf_img_features = {
        label: extract_image_features(f"{image_output_path}/{label}/mtf", gray=False)
        for label in processed_data.keys()
        if label == "DEMOGRAPHIC"
    }
    features_list.append(min_max_scaler.fit_transform(demo_mtf_img_features))

    demo_gasf_img_features = {
        label: extract_image_features(f"{image_output_path}/{label}/gasf", gray=False)
        for label in processed_data.keys()
        if label == "DEMOGRAPHIC"
    }
    features_list.append(min_max_scaler.fit_transform(demo_gasf_img_features))

    demo_gadf_img_features = {
        label: extract_image_features(f"{image_output_path}/{label}/gadf", gray=False)
        for label in processed_data.keys()
        if label == "DEMOGRAPHIC"
    }
    features_list.append(min_max_scaler.fit_transform(demo_gadf_img_features))

    # np.save("demo_image_features.npy", features_list)
    print("Image feature extraction complete.")

    np.save('/workspaces/MVC-Time-series-Forcasting/data/features/demo_All_features.npy', features_list)
    print("All features saved for DEMOGRAPHIC.")


    # Step 4: Select best views
    selected_views = {}
    for label in processed_data.keys():
        selected_views[label] = select_views(features_list, threshold=0.1)

    np.save("selected_views.npy", selected_views)
    print("View selection complete. Selected views saved.")

    # # Step 5: Perform multi-view fuzzy clustering
    # print("Running multi-view fuzzy clustering...")
    # clustering_results = fuzzy_multi_view_clustering(dataset="demo")
    # np.save("clustering_results.npy", clustering_results)
    # print("Clustering completed. Results saved.")

    # # Step 6: Run multiple experiments and collect predictions
    # print("Running experiments...")
    # combined_predictions, TestY, TrainY = run_experiments(n_runs=2, ae_name="Autoencoder_Model", dim=16)
    # np.save("combined_predictions.npy", combined_predictions)
    # print("Experiments completed. Predictions saved.")

    # # Step 7: Perform Median Ensemble Evaluation
    # print("Performing Median Ensemble Evaluation...")
    # dataset_name = "M3_Monthly"
    # mase_freq = 12  # Monthly data

    # ensemble_results = median_ensemble_evaluation(
    #     combined_predictions=combined_predictions,
    #     TestY=TestY,
    #     TrainY=TrainY,
    #     dataset_name=dataset_name,
    #     mase_freq=mase_freq,
    #     save_path="ensemble_results"
    #     )

    # print("Ensemble evaluation completed. Results saved.")
    # print("Final Ensemble Results:", ensemble_results)