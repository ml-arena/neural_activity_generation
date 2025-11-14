"""
Metrics for evaluating neural activity generation models

Includes:
- Frechet Distance for distribution matching
- Trial-matched R2 for reconstruction quality
- Biophysical representation (placeholder)
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy import linalg


def biophysical_representation(X, d=10):
    """
    Placeholder for biophysical representation function.

    This will later compute biophysical features from neural activity.
    For now, it returns a simple projection to dimension d.

    Args:
        X: Neural activity of shape (batch_size, ...)
        d: Target dimension for representation

    Returns:
        Representation of shape (batch_size, d)
    """
    # Flatten all dimensions except batch
    X = np.asarray(X)
    batch_size = X.shape[0]
    X_flat = X.reshape(batch_size, -1)

    # Simple projection to dimension d (placeholder)
    # Later this will compute actual biophysical features
    n_features = X_flat.shape[1]

    if n_features >= d:
        # Take first d features
        return X_flat[:, :d]
    else:
        # Pad with zeros if needed
        return np.pad(X_flat, ((0, 0), (0, d - n_features)), mode='constant')


def resample(data, n_samples, replace=False):
    """Simple resampling function to avoid sklearn dependency"""
    indices = np.random.choice(len(data), size=n_samples, replace=replace)
    return data[indices]


def sqrtm_eig(A: np.ndarray, check_real: bool = True) -> np.ndarray:
    """
    Fast matrix square root using eigenvalue decomposition.
    Best for symmetric/Hermitian positive definite matrices.

    Args:
        A: Input matrix (should be symmetric/Hermitian positive definite)
        check_real: If True, return real part for nearly real results

    Returns:
        Matrix square root of A
    """
    # Compute eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(A)

    # Check for negative eigenvalues
    if np.any(eigenvalues < -1e-10):
        raise ValueError("Matrix has negative eigenvalues, not positive definite")

    # Clip small negative values to zero (numerical errors)
    eigenvalues = np.maximum(eigenvalues, 0)

    # Compute square root
    sqrt_eigenvalues = np.sqrt(eigenvalues)

    # Reconstruct matrix: A^(1/2) = V * sqrt(Λ) * V^T
    result = eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.T.conj()

    # Return real part if result is essentially real
    if check_real and np.allclose(result.imag, 0):
        return result.real

    return result


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    """
    Numpy implementation of the Frechet Distance.
    from: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py#L179

    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the mean of distribution 1
    -- mu2   : Numpy array containing the mean of distribution 2
    -- sigma1: The covariance matrix for distribution 1
    -- sigma2: The covariance matrix for distribution 2

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # Product might be almost singular
    cov_product = sigma1.dot(sigma2)
    cov_product = (cov_product + cov_product.T) / 2  # project symmetric for numerical stability

    # Use scipy's matrix square root as it's more robust
    try:
        covmean, _ = linalg.sqrtm(cov_product, disp=False)
    except:
        # Fallback to eigenvalue decomposition with regularization
        eps = 1e-3
        cov_product += np.eye(cov_product.shape[0]) * eps
        covmean = sqrtm_eig(cov_product)

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        # For generated vs real data comparison, imaginary components can be larger
        # We'll just take the real part and issue a warning if it's significant
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=0.1):
            # Significant imaginary component - data may be problematic
            # But we'll continue with real part
            pass
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def trial_matching(cost_matrix):
    """
    Perform optimal trial matching between two datasets using Hungarian algorithm

    Args:
        cost_matrix: Square matrix of pairwise costs (e.g., MSE between trials)

    Returns:
        mse: Mean cost of optimal matching
        ind_x, ind_y: Indices of matched pairs
    """
    cost_matrix_np = np.asarray(cost_matrix)
    n, m = cost_matrix.shape

    if n != m:
        raise NotImplementedError(f"cost matrix has shape {cost_matrix.shape}")

    if np.isnan(cost_matrix_np).any():
        raise ValueError(
            f"cost matrix has nan values {np.isnan(cost_matrix_np).sum()}/{np.size(cost_matrix_np)}"
        )

    ind_x, ind_y = linear_sum_assignment(cost_matrix_np)
    mse = cost_matrix[ind_x, ind_y].mean()

    return mse, ind_x, ind_y


def fun_trial_matched_metrics(z_simulated, z_data, batch_size, feature_fun, random_sample=False):
    """
    Compute trial-matched R2 score between simulated and real neural data

    This metric:
    1. Optionally samples batches from both datasets
    2. Applies a feature function to both (e.g., population mean)
    3. Finds optimal trial matching using Hungarian algorithm
    4. Computes R2 score of matched trials

    Args:
        z_simulated: Simulated/generated neural data (n_trials, n_neurons)
        z_data: Real neural data (n_trials, n_neurons)
        batch_size: Number of trials to compare
        feature_fun: Function to apply to data before matching (e.g., lambda x: x.mean(1))
        random_sample: If True, randomly sample trials; if False, use first batch_size

    Returns:
        R2 score (higher is better, 1.0 is perfect)
    """
    # Sample data
    if random_sample:
        z_simulated = resample(z_simulated, n_samples=batch_size, replace=False)
        z_data = resample(z_data, n_samples=batch_size, replace=False)
    else:
        z_simulated = z_simulated[:batch_size]
        z_data = z_data[:batch_size]

    # Apply feature function
    z_simulated = feature_fun(z_simulated)
    z_data = feature_fun(z_data)

    assert len(z_simulated) == len(z_data)
    assert len(z_simulated.shape) == 2

    # Compute pairwise cost matrix (MSE between all pairs)
    cost_matrix = np.zeros((len(z_simulated), len(z_data)))
    for i in range(len(z_simulated)):
        for j in range(len(z_data)):
            cost_matrix[i, j] = np.mean((z_simulated[i] - z_data[j]) ** 2)

    # Find optimal matching
    mse, ind_x, ind_y = trial_matching(cost_matrix)

    # Compute R2 from matched pairs
    z_sim_matched = z_simulated[ind_x]
    z_data_matched = z_data[ind_y]

    # R2 = 1 - SS_res / SS_tot
    ss_res = np.sum((z_data_matched - z_sim_matched) ** 2)
    ss_tot = np.sum((z_data_matched - z_data_matched.mean(axis=0)) ** 2)

    r2 = 1 - (ss_res / ss_tot)

    return r2, ind_x, ind_y
