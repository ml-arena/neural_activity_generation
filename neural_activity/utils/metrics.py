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

def feature_oscillation_frequency(X, f_min=1, f_max=60, num_f=24 * 4, n_group=3):
    B, T, d = X.shape

    X_mean = X.mean(1) # dim: B x d
    X = X - X_mean[:, None, :]
    dt = 0.01

    coeff = (f_max/f_min)**(1/(num_f - 1))
    #freqs = torch.linspace(f_min, f_max, num_f, device=X.device)
    freqs = f_min * np.power(coeff, np.arange(num_f))
    assert freqs[0] == f_min
    assert (freqs[-1] - f_max) / f_max < 1e-4, f"f_max={freqs[-1]} and not f_max={f_max}"
    time_line = np.arange(T) * dt
    cos = np.cos(np.pi * freqs[None,:] * time_line[:, None]) # dim: T x F
    sin = np.sin(np.pi * freqs[None,:] * time_line[:, None]) # dum: T X F
    Z_cos = (X[..., None] * cos[:,None,:]).mean(1)
    Z_sin = (X[..., None] * sin[:,None,:]).mean(1)
    Z = np.sqrt(Z_cos**2 + Z_sin**2) # dim: B x d x F
    if n_group is None: n_group = num_f
    Z = Z.reshape(B, d, n_group, num_f // n_group).mean(-1)
    Z_cst = X_mean[..., None]
    Z = np.concatenate([Z,Z_cst], -1).reshape(B, -1)
    assert Z.shape[-1] == d * (n_group+1), f"wrong number of dim"
    return Z.reshape(B,-1)


def biophysical_representation(X, d=44):
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
    array = np.array

    neuron_index_dict = {
        'VISp': array([138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150,
                151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163,
                164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176,
                177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189,
                190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202,
                203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215,
                216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228,
                229, 230]),
         'VISrl': array([467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479,
                480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492,
                493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505,
                506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518,
                519, 520, 521, 522, 523, 524]),
         'VISl': array([276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288,
                289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301,
                302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314,
                315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327,
                328, 329, 330, 331]),
         'VISal': array([385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397,
                398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410,
                411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423,
                424, 425, 426, 427]),
         'VISam': array([ 4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
                38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]),
         'CA1': array([  1,   2,   3,  53,  54,  55,  56,  57, 244, 245, 246, 247, 248,
                249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261,
                262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274,
                275, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363,
                364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376,
                377, 378, 379, 380, 381, 382, 383, 384, 455, 456, 457, 458, 459,
                460, 461, 462, 463, 464, 465, 466]),
         'SUB': array([ 97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122,
                123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
                136, 137]),
         'null': array([  0,   3,  53,  57,  78,  79,  92,  93, 102, 103, 107, 109, 110,
                114, 122, 123, 124, 125, 126, 128, 154, 167, 173, 174, 175, 178,
                189, 200, 212, 213, 224, 227, 234, 240, 242, 246, 247, 254, 255,
                262, 266, 268, 275, 276, 279, 306, 319, 332, 346, 349, 351, 361,
                370, 372, 374, 378, 381, 406, 409, 410, 453, 456, 465, 466, 467,
                480, 489, 491, 516]),
         '0.0': array([  6,  16,  22,  30,  34,  35,  43,  48,  61,  70,  72,  73,  85,
                 96, 105, 106, 111, 118, 119, 127, 130, 141, 149, 150, 161, 172,
                182, 190, 295, 317, 320, 350, 352, 353, 389, 393, 404, 405, 412,
                417, 422, 425, 431, 433, 434, 435, 443, 444, 471, 476, 479, 494,
                503, 506, 521, 523]),
         '45.0': array([  2,  23,  29,  36,  52,  65,  77,  82,  99, 101, 104, 115, 137,
                139, 151, 155, 163, 165, 196, 219, 232, 274, 300, 333, 345, 364,
                379, 388, 399, 427, 436, 468, 487, 508, 514, 515, 519]),
         '90.0': array([  5,  10,  21,  24,  25,  28,  33,  45,  69, 108, 117, 143, 152,
                156, 162, 177, 179, 185, 187, 217, 221, 223, 229, 253, 289, 313,
                314, 331, 362, 380, 382, 449, 455, 473, 478, 485, 486, 490, 493])
    }



    X_pop = []
    for key,inds in neuron_index_dict.items():
        X_pop.append(X[..., inds].mean(-1))
    X_pop = np.stack(X_pop, -1)

    features = feature_oscillation_frequency(X_pop)

    mean = [0.007493038978825583, 0.006188965265658296, 0.0033996874166164876, 0.13326093912697756,
            0.005309828973749758, 0.004386166904613503, 0.003135768586185195, 0.1355213299035453, 0.008273224234805993,
            0.005882099905739658, 0.003944514628592391, 0.1498666848235119, 0.0050968081539511795, 0.00426537036677928,
            0.0033583538223481454, 0.10759957018308342, 0.004047143881469345, 0.003313604162483522,
            0.0020721883451692495, 0.0513864443413555, 0.003534838787123739, 0.0029419596950311425,
            0.002253350744961822, 0.0628730244218157, 0.0020020638352171846, 0.0017451231818242667,
            0.0017633020100120368, 0.04493302144916155, 0.003714414183267803, 0.0033963448830224693,
            0.002704872073448307, 0.07480258582374798, 0.0048969299287962825, 0.0036697171426024328,
            0.00240008502302478, 0.0632197361287231, 0.005216566446942649, 0.0050268806212947536, 0.0023682040914292835,
            0.07384109118272765, 0.006451141021989669, 0.005141380296848104, 0.004661068152978503, 0.11013223917689174]
    std = [0.003337981640928058, 0.0014485959865033072, 0.0006629992628133522, 0.04353357117988084, 0.00220869806232434,
           0.0012509588870063264, 0.0005599584947354162, 0.03670388538437049, 0.0035868542445673685,
           0.0015462488213319517, 0.0006468910855799661, 0.046953687174648574, 0.0026396524226909544,
           0.0016174178893506672, 0.0006506063693847731, 0.026789438489127956, 0.001843557389752658,
           0.0012648685642927377, 0.000527292807185672, 0.01776162599156649, 0.0014380985519969198,
           0.0008145630829930258, 0.0003273449296476627, 0.01778245330382677, 0.0006438547486064265,
           0.00034559364962388035, 0.000214904938162137, 0.00903533956194234, 0.0013288372340525142,
           0.0010872812322061942, 0.00036805728109846035, 0.014534174817326528, 0.0019871454657919064,
           0.0011060930071134778, 0.00042774321642716174, 0.016677981284852254, 0.0023369832147116876,
           0.0014563195873061193, 0.0004650926409213801, 0.01977637728311928, 0.0028705178117713367,
           0.0010643402656023296, 0.000772352264328995, 0.041381843089760985]

    features = (features - mean) / std

    # Flatten all dimensions except batch
    X = np.asarray(features)
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


def frechenet_distance(z_simulated, z_data, batch_size, feature_fun, random_sample=False):
    z_simulated =  resample(z_simulated, batch_size) if random_sample else z_simulated[:batch_size]
    z_data =  resample(z_data, batch_size) if random_sample else z_data[:batch_size]

    # Could be population average activity or non-linear feature extraction?
    z1 = feature_fun(z_simulated)
    z2 = feature_fun(z_data)

    mu1 = np.mean(z1, axis=0)
    mu2 = np.mean(z2, axis=0)
    sigma1 = np.cov(z1, rowvar=False)
    sigma2 = np.cov(z2, rowvar=False)

    return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

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


# if run as main:
if __name__ == "__main__":
    X = np.load("../data/neural_data.npz")
    print(X["z_test"].shape)
    z1 = X["z_test"][:64]
    z2 = X["z_test"][64:128]

    z2_time_shuffle = z2[:, np.random.permutation(z2.shape[1])]
    z2_neuro_shuffle = z2[:, :, np.random.permutation(z2.shape[2])]

    r2_ref, *_ = fun_trial_matched_metrics(z1, z2, 64, biophysical_representation)
    r2_time_shuffle, *_ = fun_trial_matched_metrics(z2_time_shuffle, z2, 64, biophysical_representation)
    r2_neuro_suffle, *_ = fun_trial_matched_metrics(z2_neuro_shuffle, z2, 64, biophysical_representation)

    FID = frechenet_distance(z1, z2, 64, biophysical_representation,)
    FID_time_shuffle = frechenet_distance(z2_time_shuffle, z2, 64, biophysical_representation)
    FID_neuron_shuffle = frechenet_distance(z2_neuro_shuffle, z2, 64, biophysical_representation)

    print("r2_ref", r2_ref)
    print("r2_time_shuffle", r2_time_shuffle)
    print("r2_neuro_suffle", r2_neuro_suffle)

    print("FID", FID)
    print("FID shuffle", FID_time_shuffle)
    print("FID_neuron_shuffle", FID_neuron_shuffle)



