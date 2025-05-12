import numpy as np
import matplotlib.pyplot as plt

class PCA_:
    def __init__(self, num_components):
        
        self.num_components = num_components
        self.principle_components = None
        self.mean = None
        self.variance = None
        
    def fit(self, X):
        # move origin to mean
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        ## cov(Xᵢ,Xⱼ) = Σ[(Xᵢₖ - Xᵢ_mean)(Xⱼₖ - Xⱼ_mean)] / (n-1)
        n_samples = X_centered.shape[0]
        cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)
        
        # eigenvalues and eigenvectors solve the equation : C·v = λ·v
        # where C is the covariance matrix, v is the eigenvector and λ is the eigenvalue
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # store n largest eigenvalues and corresponding eigenvectors
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]   
        self.principle_components = eigenvectors[:, :self.num_components].T
        self.variance = eigenvalues[:self.num_components]
        
        return self
    
    def transform(self, X):
        # move origin to mean and project
        X_centered = X - self.mean
        return np.dot(X_centered, self.principle_components.T)
   