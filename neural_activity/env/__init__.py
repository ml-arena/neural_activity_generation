"""Neural activity generation environment"""

from .neural_activity import NeuralActivityEnv

# Alias for consistency with other environments
Env = NeuralActivityEnv

__all__ = ['NeuralActivityEnv', 'Env']
