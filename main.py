import numpy as np
import pandas as pd
from view_generation import process_time_series
from m3_preprocessing import process_m3_dataset  # M3 dataset preprocessing
from feature_extraction import extract_time_series_features, extract_image_features  # Feature extraction


# # ================================
# #  Main Execution Script
# # ================================
# if __name__ == "__main__":
#     """
#     Unified pipeline:
#     1. Load & process M3 dataset
#     2. Generate images (RP, GAF, MTF) from time series
#     3. Extract numerical features from time series
#     4. Extract CNN features from images
#     5. Save extracted features
#     """

#     # Step 1: Process M3 dataset
#     file_path = "m3monthdataset/M3Month.csv"
#     processed_data = process_m3_dataset(file_path)

#     if processed_data:
#         print("✅ Time-series processing complete. Categories:", list(processed_data.keys()))

#         # Step 2: Generate images from time series
#         image_output_path = "generated_images/"
#         for label, ts_data in processed_data.items():
#             print(f"🔄 Generating images for category: {label}")
#             df = pd.DataFrame(ts_data)  # Convert to DataFrame for processing
#             process_time_series(df, output_path=f"{image_output_path}/{label}")

#         # Step 3: Extract numerical features from time series
#         ts_features = {label: extract_time_series_features(data) for label, data in processed_data.items()}
#         np.save("time_series_features.npy", ts_features)
#         print("✅ Time-series feature extraction complete. Features saved.")

#         # Step 4: Extract image-based CNN features
#         img_features = {label: extract_image_features(f"{image_output_path}/{label}", gray=True) for label in processed_data.keys()}
#         np.save("image_features.npy", img_features)
#         print("✅ Image feature extraction complete. Features saved.")

#     else:
#         print("❌ Error in processing M3 dataset. Please check file path and dataset format.")
