import numpy as np

def mahalanobis_distance(data, point) -> float:
    """
    Calculate the Mahalanobis distance of a point from a dataset.
    
    Parameters:
    data (np.array): A 2D array where each row is a data point.
    point (np.array): A 1D array representing the point to calculate the distance for.
    
    Returns:
    float: The Mahalanobis distance of the point from the dataset.
    """
    # Calculate the mean of the dataset
    mean = np.mean(data, axis=0)
    
    # Calculate the covariance matrix of the dataset
    cov_matrix = np.cov(data, rowvar=False)
    
    # Calculate the inverse of the covariance matrix
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    
    # Calculate the Mahalanobis distance
    diff = point - mean
    mahalanobis_dist = np.sqrt(diff.T @ inv_cov_matrix @ diff)
    
    return mahalanobis_dist
