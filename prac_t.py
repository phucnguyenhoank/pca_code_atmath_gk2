import numpy as np
import matplotlib.pyplot as plt

mean = [0, 0]
cov = [[2, 3], 
       [3, 5]]

X, y = np.random.multivariate_normal(mean, cov, 100).T

X_mean = np.mean(X, axis=1, keepdims=True)  # Mean of each feature, shape: (2,1)
X_std = np.std(X, axis=1, keepdims=True)     # Std of each feature, shape: (2,1)
X_normalized = (X - X_mean) / X_std            # Standardized data

cov_matrix = X_normalized @ X_normalized.T / (X.shape[1])  # Shape: (2,2)


