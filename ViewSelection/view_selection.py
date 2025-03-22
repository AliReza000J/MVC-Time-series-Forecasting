import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import kneighbors_graph

def compute_diversity(feature_list):
    """
    Compute diversity weights for different views.
    
    Parameters:
        feature_list (list): List of extracted feature matrices.
    
    Returns:
        np.array: Normalized diversity weights.
    """
    min_max_scaler = MinMaxScaler()
    trace_matrix = np.zeros((len(feature_list), len(feature_list)))

    for i in range(len(feature_list)):
        for j in range(len(feature_list)):
            arr = np.dot(
                min_max_scaler.fit_transform(feature_list[i]),
                min_max_scaler.fit_transform(feature_list[j]).T
            )
            trace_matrix[i, j] = arr.trace()

    upper_triangular = np.triu(trace_matrix)
    row_sums = upper_triangular.sum(axis=1)
    row_sums[row_sums == 0] = np.finfo(float).eps
    normalized_row_sums = (1 / row_sums) / np.sum(1 / row_sums)

    return normalized_row_sums

def compute_similarity(feature_list, k=3):
    """
    Compute similarity weights using manifold constraint.
    
    Parameters:
        feature_list (list): List of extracted feature matrices.
        k (int): Number of neighbors for kNN graph.
    
    Returns:
        np.array: Normalized similarity weights.
    """
    min_max_scaler = MinMaxScaler()
    sim = np.zeros(len(feature_list))

    for i in range(len(feature_list)):
        knn_graph = kneighbors_graph(
            min_max_scaler.fit_transform(feature_list[i]), k, mode='distance', include_self=True
        )
        laplacian = np.diag(np.sum(knn_graph.toarray(), axis=1)) - knn_graph
        quadratic_form = np.dot(np.dot(laplacian, feature_list[i]), feature_list[i].T)
        sim[i] = np.abs(quadratic_form.trace())

    inverted_sim = 1 / sim
    normalized_sim = inverted_sim / inverted_sim.sum()

    return normalized_sim

def select_views(feature_list, threshold=0.1):
    """
    Select the most representative views based on diversity and similarity.
    
    Parameters:
        feature_list (list): List of extracted feature matrices.
        threshold (float): Minimum weight threshold for selection.
    
    Returns:
        list: Indices of selected views.
    """
    W_div = compute_diversity(feature_list)
    W_sim = compute_similarity(feature_list)

    W = np.multiply(W_div, W_sim.T)
    normalized_W = W / W.sum()

    remaining_indices = np.where(normalized_W >= threshold)[0]
    return remaining_indices