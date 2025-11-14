"""
Neural Activity Generation Environment for Supervised Learning

This environment evaluates generative models of neural activity using:
1. Reconstruction quality (R2 via trial-matched metrics)
2. Generation quality (Frechet Distance on generated samples)

The agent must implement:
- encode(X) -> embeddings
- decode(z) -> reconstructed X
"""
import numpy as np
import os
from typing import Optional, Dict, Any
from neural_activity import PKG_DIR
from neural_activity.utils import calculate_frechet_distance, fun_trial_matched_metrics, biophysical_representation


class NeuralActivityEnv:
    """
    Environment for evaluating neural activity generation models

    The environment provides batches of neural activity data and evaluates
    agent performance on two tasks:
    1. Reconstruction: encode then decode the data, measure quality with R2
    2. Generation: generate new samples from noise, measure quality with Frechet Distance
    """

    def __init__(
        self,
        batch_size: int = 100,
        dataset_path: Optional[str] = None,
        dataset: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the neural activity environment

        Args:
            batch_size: Number of trials to use for evaluation
            dataset_path: Optional path to custom dataset .npz file
            dataset: Optional pre-loaded dataset dictionary with 'z_test' key
        """
        self.batch_size = batch_size
        self.current_batch = 0
        self.rng = np.random.RandomState()

        # Load neural activity dataset
        if dataset is not None:
            # Use pre-loaded dataset
            self.z_data = dataset['z_test']
        elif dataset_path is not None:
            # Load from custom path
            try:
                data = np.load(dataset_path, allow_pickle=True)
                if isinstance(data, np.ndarray) and data.shape == ():
                    # It's a .npy file with a dictionary
                    data = data.item()
                    self.z_data = data.get('z_test', data.get('data'))
                else:
                    # It's a .npz file
                    self.z_data = data['z_test']
            except (FileNotFoundError, OSError, KeyError) as e:
                raise RuntimeError(
                    f"Neural activity dataset not found at: {dataset_path}"
                ) from e
        else:
            # Load default dataset
            data_path = os.path.join(PKG_DIR, 'data')
            try:
                data = np.load(os.path.join(data_path, 'neural_data.npz'))
                self.z_data = data['z_test']
            except (FileNotFoundError, OSError, KeyError) as e:
                raise RuntimeError(
                    "Default neural activity dataset not found. Please ensure the package is installed correctly."
                ) from e

        # Validate dataset
        if len(self.z_data.shape) == 2:
            # Old format: (n_trials, n_neurons)
            self.n_trials, self.n_neurons = self.z_data.shape
            self.n_time = 1
        elif len(self.z_data.shape) == 3:
            # New format: (n_trials, n_time, n_neurons)
            self.n_trials, self.n_time, self.n_neurons = self.z_data.shape
        else:
            raise RuntimeError(f"Invalid dataset shape: {self.z_data.shape}. Expected (n_trials, n_neurons) or (n_trials, n_time, n_neurons)")

        if self.n_trials == 0:
            raise RuntimeError("Dataset is empty")

        # We only have one task per simulation (evaluate the entire dataset)
        self.num_batches = 1

        # Track current task data
        self.current_test_data = None

    def set_seed(self, seed: Optional[int] = None):
        """Set random seed for reproducibility"""
        self.rng = np.random.RandomState(seed)

    def get_next_task(self) -> Optional[Dict[str, Any]]:
        """
        Get the next evaluation task

        Returns:
            Dictionary with:
                - X: Neural activity data for encoding (n_trials, n_neurons)
                - Z: Latent representations to decode for generation (n_trials, n_neurons)
            Returns None when all batches are complete
        """
        if self.current_batch >= self.num_batches:
            return None

        # Use all data or a batch
        if self.batch_size >= self.n_trials:
            test_data = self.z_data
        else:
            # Sample batch_size trials
            indices = self.rng.choice(self.n_trials, size=self.batch_size, replace=False)
            test_data = self.z_data[indices]

        # Store for evaluation
        self.current_test_data = test_data

        self.current_batch += 1

        # For encode/decode workflow:
        # X = neural activity to encode -> Z_pred -> Y_pred (reconstruction)
        # Z = latent codes to decode -> Y_gen (generation)
        # For simplicity, we use the same data for both tasks
        return {
            'X': test_data,  # Neural activity to encode/decode
            'Z': test_data   # Latent representations to decode for generation
        }

    def evaluate(self, Z_pred: np.ndarray, Y_pred: np.ndarray, Y_gen: np.ndarray, d: int = 10) -> tuple:
        """
        Evaluate agent performance on reconstruction and generation

        This follows the encode/decode workflow:
        1. Agent encodes X -> Z_pred
        2. Agent decodes Z_pred -> Y_pred (reconstruction)
        3. Agent decodes Z (true latents) -> Y_gen (generation)

        Args:
            Z_pred: Predicted latent embeddings from encoding X
            Y_pred: Reconstructed neural activity from decoding Z_pred
            Y_gen: Generated neural activity from decoding true latents Z
            d: Dimension for biophysical representation

        Returns:
            Tuple of (metric, metric2):
                - metric (r2_reconstruction): R2 score for reconstruction (higher is better)
                - metric2 (fid_generation): Frechet Distance for generation (lower is better)
        """
        Z_pred = np.asarray(Z_pred)
        Y_pred = np.asarray(Y_pred)
        Y_gen = np.asarray(Y_gen)

        # Use stored test data as ground truth
        X_test = self.current_test_data
        if X_test is None:
            raise RuntimeError("No current test data available. Call get_next_task() first.")

        # Validate shapes
        if Y_pred.shape != X_test.shape:
            raise ValueError(
                f"Reconstructed shape {Y_pred.shape} doesn't match data shape {X_test.shape}"
            )

        # Task 1: Reconstruction quality (R2 via trial-matched metrics)
        # Apply biophysical representation
        X_test_repr = biophysical_representation(X_test, d=d)
        Y_pred_repr = biophysical_representation(Y_pred, d=d)

        # Use biophysical representation as feature
        feature_fun = lambda x: biophysical_representation(x, d=d)

        eval_batch_size = min(self.batch_size, len(X_test), len(Y_pred))

        try:
            r2_score, _, _ = fun_trial_matched_metrics(
                Y_pred, X_test, eval_batch_size, feature_fun, random_sample=False
            )
        except Exception as e:
            print(f"Warning: R2 computation failed: {e}")
            r2_score = -1.0  # Bad score on failure

        # Task 2: Generation quality (Frechet Distance)
        # Apply biophysical representation to both real and generated
        X_test_repr_fid = biophysical_representation(X_test, d=d)
        Y_gen_repr = biophysical_representation(Y_gen, d=d)

        try:
            # Use a subset for FID computation if datasets are large
            fid_batch_size = min(self.batch_size, len(X_test_repr_fid), len(Y_gen_repr))

            Y_gen_sample = Y_gen_repr[:fid_batch_size]
            X_real_sample = X_test_repr_fid[:fid_batch_size]

            # Compute statistics in biophysical representation space
            mu_gen = np.mean(Y_gen_sample, axis=0)
            mu_real = np.mean(X_real_sample, axis=0)

            # Add strong regularization to ensure positive definiteness
            eps = 1e-3
            sigma_gen = np.cov(Y_gen_sample, rowvar=False) + np.eye(d) * eps
            sigma_real = np.cov(X_real_sample, rowvar=False) + np.eye(d) * eps

            fid_distance = calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)

        except Exception as e:
            print(f"Warning: FID computation failed: {e}")
            fid_distance = float('inf')  # Bad score on failure

        # Return both metrics as tuple (metric, metric2)
        # metric = reconstruction quality (higher is better)
        # metric2 = generation quality via Frechet distance (lower is better)
        return (float(r2_score), float(fid_distance))

    def reset(self):
        """Reset environment to start from first batch"""
        self.current_batch = 0
        self.current_test_data = None

    def is_complete(self) -> bool:
        """Check if all batches are complete"""
        return self.current_batch >= self.num_batches
