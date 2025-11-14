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
    - encode: Returns the input (identity mapping)
    - decode: Returns the input (identity mapping)
    - Generates random samples from Gaussian matching data statistics
    """

    def __init__(self):
        """Initialize the naive agent"""
        self.fitted = False
        self.mean = None
        self.std = None
        self.n_neurons = None

    def reset(self):
        """Reset agent state"""
        self.fitted = False
        self.mean = None
        self.std = None
        self.n_neurons = None

    def encode(self, X):
        """
        Encode neural activity to latent space

        For the naive agent, this is just identity mapping

        Args:
            X: Neural activity data (n_trials, n_neurons)

        Returns:
            Embeddings (same as input)
        """
        return np.asarray(X)

    def decode(self, z):
        """
        Decode from latent space to neural activity

        For the naive agent, this is just identity mapping

        Args:
            z: Latent embeddings (n_samples, latent_dim)

        Returns:
            Reconstructed neural activity (same as input)
        """
        return np.asarray(z)

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
