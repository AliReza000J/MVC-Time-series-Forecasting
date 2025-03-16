import numpy as np
from sklearn.metrics import silhouette_score

# Load data and labels from .npy files
# data = np.load('data_file.npy')  # replace with your data file path
labels = np.load('labels_file.npy')  # replace with your labels file path
mat = np.load('dataset/All_features_16/MICRO_All_features_16.npy')
data = {}
print(mat[4])
# print(len(mat))
data[0] = mat[0]
data[1] = mat[2]
data[2] = mat[3]
data[3] = mat[4]
# data[4] = mat[4]
# Compute Silhouette score
score = silhouette_score(data, labels)
print(f'Silhouette Score: {score}')