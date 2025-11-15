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

    def reset(self):
        """Reset agent state"""
        self.fitted = False
        self.mean = None
        self.std = None
        self.n_neurons = None
        self.encode_matrix = None
        self.decode_matrix = None

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
            X: Neural activity data (n_trials, n_neurons)

        Returns:
            Latent embeddings (n_trials, 4)
        """
        X = np.asarray(X)

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
            Reconstructed neural activity (n_samples, n_neurons)
        """
        z = np.asarray(z)

        # Need to initialize if decode is called before encode
        if self.decode_matrix is None:
            # Use a default neuron count (will be set properly on first encode)
            self._initialize_matrices(40)  # Default for this dataset

        # Project back to neural space
        X_reconstructed = z @ self.decode_matrix

        # Ensure non-negative (neural firing rates)
        X_reconstructed = np.maximum(X_reconstructed, 0)

        return X_reconstructed.astype(np.float32)

    def predict(self, X_test):
        """
        Generate predictions for the environment

        This method is called by the REST API and should return
        both reconstructed and generated samples.

        Args:
            X_test: Neural activity data to evaluate on
                   Shape: (n_trials, n_neurons) or (n_trials, n_time, n_neurons)

        Returns:
            Dictionary with 'reconstructed' and 'generated' keys
        """
        X_test = np.asarray(X_test)

        # Fit statistics on test data (for generation)
        if not self.fitted:
            # Compute statistics across all dimensions except last (neurons)
            self.mean = np.mean(X_test, axis=tuple(range(len(X_test.shape) - 1)))
            self.std = np.std(X_test, axis=tuple(range(len(X_test.shape) - 1))) + 1e-6
            self.data_shape = X_test.shape
            self.fitted = True

        # Reconstruction: encode then decode
        embeddings = self.encode(X_test)
        reconstructed = self.decode(embeddings)

        # Generation: sample from Gaussian matching data statistics and shape
        generated = np.random.randn(*X_test.shape) * self.std + self.mean

        # Ensure non-negative (neural firing rates)
        generated = np.maximum(generated, 0)

        return {
            'reconstructed': reconstructed,
            'generated': generated
        }
