# Neural Activity Generation Environment

An encode/decode environment for evaluating generative models of neural population activity.

## Overview

Build models that can both reconstruct and generate realistic neural population activity. Your agent must learn to encode neural activity into latent representations and decode them back to generate new activity patterns.

## Competition Submission Requirements

Your submission must include:
- **File:** `agent.py`
- **Class:** `Agent`
- **Methods:**
  - `def encode(self, X: np.ndarray) -> np.ndarray` - Encode neural activity to latents
  - `def decode(self, z: np.ndarray) -> np.ndarray` - Decode latents to neural activity

You can include additional methods, import files, or `.pt` weight files as needed.

## Agent Example

Your `agent.py` with encode/decode methods:

```python
import numpy as np
import torch
import torch.nn as nn

class Agent:
    def __init__(self):
        # Load your pre-trained encoder/decoder
        self.encoder = torch.load('encoder.pt')
        self.decoder = torch.load('decoder.pt')
        self.encoder.eval()
        self.decoder.eval()

    def reset(self):
        pass  # Optional: reset agent state

    def encode(self, X):
        """
        Encode neural activity to latent representation

        Args:
            X: Neural activity (n_trials, n_neurons)

        Returns:
            Latent embeddings (n_trials, latent_dim)
        """
        X_tensor = torch.FloatTensor(X)

        with torch.no_grad():
            z = self.encoder(X_tensor)

        return z.numpy()

    def decode(self, z):
        """
        Decode latent representation to neural activity

        Args:
            z: Latent embeddings (n_samples, latent_dim)

        Returns:
            Neural activity (n_samples, n_neurons)
        """
        z_tensor = torch.FloatTensor(z)

        with torch.no_grad():
            X_reconstructed = self.decoder(z_tensor)

        return X_reconstructed.numpy()
```

## Evaluation

Your model is evaluated on two tasks:

### 1. Reconstruction Quality (Metric)
- **Measured by:** R² score using trial-matched metrics
- **Range:** -∞ to 1.0 (higher is better)
- **Task:** Agent encodes X → Z_pred, then decodes Z_pred → Y_pred
- **Score:** How well Y_pred matches original X

### 2. Generation Quality (Metric2)
- **Measured by:** Fréchet Distance in biophysical representation space
- **Range:** 0 to ∞ (lower is better)
- **Task:** Agent decodes true latents Z → Y_gen
- **Score:** Statistical similarity between generated and real neural activity

**Both metrics are computed using a biophysical representation** that captures realistic neural dynamics.

## Evaluate Locally

Test your agent before submission:

```python
from neural_activity import NeuralActivityEnv
from agent import Agent  # Your agent implementation

# Initialize environment and agent
env = NeuralActivityEnv(batch_size=100)
agent = Agent()

# Run evaluation
task = env.get_next_task()
if task:
    X = task['X']  # Neural activity to encode
    Z = task['Z']  # True latents to decode

    # Agent workflow
    Z_pred = agent.encode(X)              # Encode X to latents
    Y_pred = agent.decode(Z_pred)         # Decode predicted latents (reconstruction)
    Y_gen = agent.decode(Z)               # Decode true latents (generation)

    # Evaluate both tasks
    r2_score, fid = env.evaluate(Z_pred, Y_pred, Y_gen)

    print(f"Reconstruction R²: {r2_score:.4f}")
    print(f"Generation FID: {fid:.4f}")
```

## Dataset

- **Source:** Allen Institute Neuropixels (VISp, session 791319847)
- **Format:** `(n_trials, n_neurons)` - 40 neurons, ~140 trials
- **Task:** Learn to model the statistical structure of neural population activity

## Tips for Success

1. **Good Reconstruction:**
   - Preserve temporal dynamics
   - Maintain neuron-to-neuron correlations
   - Use appropriate loss functions (e.g., MSE, Poisson)

2. **Good Generation:**
   - Match population statistics (mean, covariance)
   - Generate diverse but realistic patterns
   - Consider variational approaches (VAE, flow models)

3. **Architecture Ideas:**
   - Variational Autoencoders (VAE)
   - Normalizing Flows
   - Transformer-based models
   - Recurrent networks for temporal data

## Installation

```bash
pip install -e .
```

## Testing

```bash
poetry run python test_package.py
```

## Versioning

Current version: **0.1**

**Changelog:**
- v0.1: Initial release with dual-task evaluation (R² + Fréchet Distance)
