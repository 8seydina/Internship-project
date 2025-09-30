import numpy as np
from scipy.linalg import cholesky
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

def generated(n_sample):
    "Dataset generated with Gaussian Distribution"

    # Set random seed for reproducibility
    np.random.seed(42)

    # Parameters for the Gaussians
    mu_A = np.array([2, 2])      # Mean for Cluster A
    Sigma_A = np.array([[1, 0],  # Covariance for Cluster A
                    [0, 1]])

    mu_B = np.array([6, 6])      # Mean for Cluster B
    Sigma_B = np.array([[1, 0],  # Covariance for Cluster B
                    [0, 1]])

    # Compute Cholesky decompositions
    L_A = cholesky(Sigma_A, lower=True)  # L_A @ L_A.T = Sigma_A
    L_B = cholesky(Sigma_B, lower=True)  # L_B @ L_B.T = Sigma_B

    # Number of samples per cluster
    n_samples = 100

    # Generate epsilon ~ N(0, I)
    epsilon = np.random.normal(0, 1, size=(2, n_samples))

    # Generate clusters using x = mu + L @ epsilon
    cluster_A = mu_A[:, np.newaxis] + L_A @ epsilon
    cluster_B = mu_B[:, np.newaxis] + L_B @ epsilon

    # Combine into a single dataset (shape: (200, 2))
    X = np.hstack([cluster_A, cluster_B]).T
    Y = np.array([-1]*n_samples + [1]*n_samples) 
    return X, Y

def  existing(n_samples):
    x, y = make_moons(n_samples)
    return x, y

def SplitDataset(x, y, test_size=0.75):
    # Semi-supervised split: small portion labeled
    x_labeled, x_unlabeled, y_labeled, _ = train_test_split(x, y, test_size=test_size, stratify=y, random_state=42)
    return x_labeled, x_unlabeled, y_labeled




