"""
Naive baseline agent for neural activity generation

This agent provides a simple baseline using:
- Identity encoding/decoding (returns input as-is)
- Random generation from a simple Gaussian
"""
import numpy as np


class Agent:
    """
    Naive baseline agent for neural activity generation

    This agent implements minimal encode/decode functionality:
    - encode: Simple linear projection to 4D latent space
    - decode: Linear projection back to neural space
    - Generates random samples from Gaussian matching data statistics
    """

    def __init__(self):
        """Initialize the naive agent"""
        self.latent_dim = 4  # Fixed latent dimension
        self.fitted = False
        self.mean = None
        self.std = None
        self.n_neurons = None
        self.encode_matrix = None
        self.decode_matrix = None
        self.original_shape = None  # Store original shape for reshaping decode output

    def reset(self):
        """Reset agent state"""
        self.fitted = False
        self.mean = None
        self.std = None
        self.n_neurons = None
        self.encode_matrix = None
        self.decode_matrix = None
        self.original_shape = None

    def _initialize_matrices(self, n_neurons):
        """Initialize random projection matrices"""
        if self.encode_matrix is None:
            # Random projection to compress to latent_dim
            self.encode_matrix = np.random.randn(n_neurons, self.latent_dim) * 0.1
            # Random projection to expand back to n_neurons
            self.decode_matrix = np.random.randn(self.latent_dim, n_neurons) * 0.1
            self.n_neurons = n_neurons

    def encode(self, X):
        """
        Encode neural activity to latent space

        For the naive agent, this uses a simple random linear projection

        Args:
            X: Neural activity data (n_trials, n_neurons) or (n_trials, n_time, n_neurons)

        Returns:
            Latent embeddings (n_trials, 4)
        """
        X = np.asarray(X)

        # Store original shape for decode reshaping
        self.original_shape = X.shape

        # Flatten if needed (handle temporal data)
        if len(X.shape) > 2:
            batch_size = X.shape[0]
            X = X.reshape(batch_size, -1)

        # Initialize projection matrices
        self._initialize_matrices(X.shape[1])

        # Project to latent space
        z = X @ self.encode_matrix
        return z.astype(np.float32)

    def decode(self, z):
        """
        Decode from latent space to neural activity

        For the naive agent, this uses a simple random linear projection

        Args:
            z: Latent embeddings (n_samples, 4)

        Returns:
            Reconstructed neural activity - shape matches original input to encode()
            (n_samples, n_neurons) or (n_samples, n_time, n_neurons)
        """
        z = np.asarray(z)

        # Need to initialize if decode is called before encode
        if self.decode_matrix is None:
            # Use a default neuron count (will be set properly on first encode)
            self._initialize_matrices(105000)  # Default for 200 timesteps × 525 neurons

        # Project back to neural space
        X_reconstructed = z @ self.decode_matrix

        # Ensure non-negative (neural firing rates)
        X_reconstructed = np.maximum(X_reconstructed, 0)

        # Reshape back to original shape if we have it
        if self.original_shape is not None and len(self.original_shape) > 2:
            batch_size = z.shape[0]
            X_reconstructed = X_reconstructed.reshape(batch_size, *self.original_shape[1:])

        return X_reconstructed.astype(np.float32)