import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math
from numpy import linalg
import scipy as sc

def causal_conv1d(x, weight, bias=None, stride=1, dilation=1, groups=1, time_dim=1, padding_val=0.):
    """
    Causal 1D convolution function.

    Args:
        x: Input tensor of shape (batch_size, in_channels, length)
        weight: Convolution weight of shape (out_channels, in_channels, kernel_size)
        bias: Optional bias tensor of shape (out_channels,)
        stride: Convolution stride (default: 1)
        dilation: Dilation factor (default: 1)

    Returns:
        Output tensor with causal convolution applied
    """
    if time_dim == 1:
        return causal_conv1d(x.permute([0,2,1]), weight, bias, stride, dilation, groups, time_dim=2).permute([0,2,1])
    elif time_dim == 0:
        return causal_conv1d(x.permute([1,0,2]), weight, bias, stride, dilation, groups, time_dim=2).permute([1,0,2])

    batch_size, in_channels, length = x.shape
    out_channels, _, kernel_size = weight.shape

    # Calculate padding needed for causal convolution
    # For causal conv, we pad only on the left side
    padding = (kernel_size - 1) * dilation

    # Pad the input on the left side only
    if padding == 0:
        x_padded = F.pad(x, (padding, 0))
    elif isinstance(padding_val, float):
        padding = torch.ones((batch_size, in_channels, padding), device=x.device) * padding_val
        x_padded = torch.cat([padding, x], time_dim)
    else:
        assert padding_val.shape == [batch_size, in_channels, padding], f"padding tensor of shape={padding_val.shape}"
        x_padded = torch.cat([padding_val, x], time_dim)

    # Apply standard convolution
    output = F.conv1d(x_padded, weight, bias=bias, stride=stride, dilation=dilation, groups=groups)

    return output

def has_nan(tensor):
    return torch.isnan(tensor).any()

def nan_count(tensor):
    return torch.isnan(tensor).int().sum()

def smooth(x, tau, dt, time_dim=1, padding=True):
    if isinstance(x, np.ndarray):
        return smooth(torch.tensor(x, dtype=torch.float), tau, dt, time_dim).detach().cpu().numpy()

    if len(x.shape) == 1:
        return smooth(x[None, :, None], tau, dt)[0,:,0]
    if len(x.shape) == 2:
        return smooth(x[:, :, None], tau, dt)[:,:,0]
    if time_dim == 1:
        return smooth(x.permute([0,2,1]), tau, dt, time_dim=2).permute(0,2,1)
    assert time_dim == 2
    B, C, T = x.shape
    x = x.reshape([B * C, 1, T])

    k = int(tau / dt)
    filter = 1 - torch.abs(torch.arange(2*k+1, device=x.device) - k) / k
    filter = filter[None, None, :] / filter.sum()
    x_mean = x.mean(0, keepdim=True)
    if padding:
        x_prepend = x[... , :k].mean(-1, keepdim=True).repeat((1,1,k))
        x_append = x[... , -k:].mean(-1, keepdim=True).repeat((1,1,k))
        x = torch.cat([x_prepend, x, x_append], time_dim)
    #x = F.pad(x, (k, k)) #+ x_mean
    out = F.conv1d(x, filter)
    return out.view(B, C, T)

class MyBatchNorm1d(nn.Module):

    def __init__(self, num_channels):
        super(MyBatchNorm1d, self).__init__()
        self.bn = nn.BatchNorm1d(num_channels)

    def forward(self, x):
        if len(x.shape) in [1,2]: return self.bn(x)
        if len(x.shape) == 3:
            return self.bn(x.permute([0,2,1])).permute([0,2,1])
        raise NotImplementedError()

class MyScaler(nn.Module):
    def __init__(self, num_channels):
        super(MyScaler, self).__init__()

        self.m = nn.Parameter(torch.zeros(num_channels))
        self.s = nn.Parameter(torch.ones(num_channels))

    def forward(self, x):
        return self.s * x + self.m

class RMSNorm(nn.Module):
    def __init__(self, num_channels, epsi=1e-6):
        super().__init__()
        self.epsi = epsi
        self.scalar = MyScaler(num_channels)

    def forward(self, x):
        return self.scalar( F.normalize(x, p=2, dim=-1, eps=self.epsi))

def create_gaussian_peaks(n_basis, duration, dt):
    """
    Create causal Fourier basis (cosine + sine) for GLM.
    Good for capturing oscillatory components in neural responses.

    Args:
        n_basis: Number of basis functions (should be even for cos/sin pairs)
        duration: Duration of the basis functions (in seconds)
        dt: Time step (in seconds)
        max_freq: Maximum frequency (Hz). If None, uses Nyquist/4

    Returns:
        basis: Tensor of shape (n_basis, n_time_points)
        time: Time vector
    """
    n_time = int(duration / dt)
    time = torch.arange(0, duration, dt)

    # Create frequency vector
    p = 2
    peak_times = torch.linspace(0, (duration /2)**(1/p), n_basis)**p

    basis = torch.zeros(n_basis, n_time)

    # Add cosine and sine components
    for i, t0 in enumerate(peak_times):
        sigma2 = torch.maximum(t0**3, (2*dt)**2)
        # Cosine basis
        basis[i] = torch.exp(- (time - t0)**2 / (2*sigma2))


    # Normalize each basis function
    for i in range(n_basis):
        if torch.norm(basis[i]) > 0:
            basis[i] = basis[i] / torch.norm(basis[i])

    return basis, time


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, T=5000):
        super().__init__()
        self.d = d_model
        pe = torch.zeros(T, d_model)
        position = torch.arange(0, T).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(T*2) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, t0, tend):
        return self.pe[t0:tend, :]


def gan_loss(z_simu, z_data, discriminator, reversal_lambda=1.0):

    # trick to implement gradient reversal:
    z_simu_rev = - reversal_lambda * z_simu
    z_simu_with_reversal = (z_simu - z_simu_rev).detach() + z_simu_rev # use

    return discriminator(z_simu_with_reversal, z_data)


def modify_optimizer_lr(optimizer, new_lr):
    # Method 1: Directly modify all parameter groups
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr  # new learning rate


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
    #if np.any(eigenvalues < -1e-10):
    #    raise ValueError("Matrix has negative eigenvalues, not positive definite")

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

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    from: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py#L179
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    cov_product = sigma1.dot(sigma2)
    cov_product = (cov_product + cov_product.T) / 2 # proj symmetric for numerical stability?
    covmean = sqrtm_eig(cov_product) #, disp=False)


    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


class LowRankWeightMatrix(nn.Module):

    def __init__(self, n1, n2, K):
        super().__init__()

        self.K = K
        self.U = nn.Parameter(torch.randn(n1,K) / np.sqrt(n1))
        self.V = nn.Parameter(torch.randn(K,n2) / np.sqrt(K))
        self.scale = nn.Parameter(torch.ones(1))# *  np.sqrt(K / n))
        self.forward()

    def forward(self,):
        if self.training: self.scale.data = torch.relu(self.scale.data)
        return self.U @ (self.V * self.scale)


class LowRankLinear(nn.Module):

    def __init__(self, n1, n2, K):
        super().__init__()
        self.W = LowRankWeightMatrix(n1, n2, K)
        self.bias = nn.Parameter(torch.zeros(n2))

    def forward(self, x):
        return x @ self.W() + self.bias