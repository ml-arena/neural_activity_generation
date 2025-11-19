"""
Test script for neural activity generation package

This tests the environment and naive agent to ensure everything works correctly.
"""
import numpy as np
import sys
import os
from test_lstm_agent import MyAgent

# Add package to path
sys.path.insert(0, os.path.dirname(__file__))

from neural_activity.env import Env as NeuralActivityEnv
from neural_activity.agent.naive import Agent

def test_environment():
    """Test the neural activity environment"""
    print("=" * 60)
    print("Testing Neural Activity Environment")
    print("=" * 60)

    # Create environment
    env = NeuralActivityEnv(batch_size=50)
    print(f"✓ Environment created")

    # Reset
    env.reset()
    print(f"✓ Environment reset")

    # Get task
    task = env.get_next_task()
    assert task is not None, "Should get a task"
    assert 'X_test' in task, "Task should have X_test"
    assert 'y_test' in task, "Task should have y_test"
    print(f"✓ Got task with shape: {task['X_test'].shape}")

    # Test completion
    is_complete = env.is_complete()
    print(f"✓ Is complete: {is_complete}")

    return env, task


def test_agent(env, task):
    """Test the naive agent"""
    print("\n" + "=" * 60)
    print("Testing Naive Agent")
    print("=" * 60)

    # Create agent
    agent = Agent()
    print("✓ Agent created")

    # Get predictions
    X_test = task['X_test']
    predictions = agent.predict(X_test)

    assert isinstance(predictions, dict), "Predictions should be a dictionary"
    assert 'reconstructed' in predictions, "Should have reconstructed"
    assert 'generated' in predictions, "Should have generated"
    print(f"✓ Got predictions:")
    print(f"  - Reconstructed shape: {predictions['reconstructed'].shape}")
    print(f"  - Generated shape: {predictions['generated'].shape}")

    return agent, predictions

def test_lstm_agent(env, task):
    # Submit the agent here to compete with your peers
    # https://ml-arena.com/viewcompetition/20
    # In the agent.py of the competition: Copy-paste the following 3 part
    # 1) Copy-paste imports and constants (T=200, d_latent=4, ...)

    # 2) Copy-paste your VAE class and related objet (ResLstm, Vec2Spikes, VAE)

    # 3) Copy-paste the Agent class with the competition API


    # End of copy-paste, small test below:
    competitionAgent = MyAgent()

    # Get predictions
    X_test = task['X_test']
    latent = competitionAgent.encode(X_test)
    latent_gen = np.random.randn((latent.shape))

    X_recon = competitionAgent.decode(latent)
    X_gen = competitionAgent.decode(latent_gen)

    predictions = {
        'reconstructed': X_recon,
        'generated': X_gen
    }


    assert isinstance(predictions, dict), "Predictions should be a dictionary"
    assert 'reconstructed' in predictions, "Should have reconstructed"
    assert 'generated' in predictions, "Should have generated"
    print(f"✓ Got predictions:")
    print(f"  - Reconstructed shape: {predictions['reconstructed'].shape}")
    print(f"  - Generated shape: {predictions['generated'].shape}")


def test_evaluation(env, task, predictions):
    """Test evaluation"""
    print("\n" + "=" * 60)
    print("Testing Evaluation")
    print("=" * 60)

    # Evaluate with new signature
    scores = env.evaluate(
        X_test=task['y_test'],
        X_pred=predictions['reconstructed'],
        X_generated=predictions['generated']
    )

    print(f"✓ R2 reconstruction: {scores['r2_reconstruction']:.4f}")
    print(f"✓ FID generation: {scores['fid_generation']:.4f}")

    assert isinstance(scores, dict), "Scores should be a dictionary"
    assert 'r2_reconstruction' in scores, "Should have r2_reconstruction"
    assert 'fid_generation' in scores, "Should have fid_generation"

    return scores


def test_full_workflow():
    """Test the complete workflow"""
    print("\n" + "=" * 60)
    print("Testing Full Workflow")
    print("=" * 60)

    # Create environment and agent
    env = NeuralActivityEnv(batch_size=50)
    agent = MyAgent()
    env.reset()

    scores = []
    task_count = 0

    while not env.is_complete():
        # Get task
        task = env.get_next_task()
        if task is None:
            break

        task_count += 1
        print(f"\nTask {task_count}:")

        # Get predictions
        #predictions = agent.predict(task['X_test'])

        latent = agent.encode(task['X_test'])
        latent_gen = np.random.randn((latent.shape))

        X_recon = agent.decode(latent)
        X_gen = agent.decode(latent_gen)

        # Evaluate
        score = env.evaluate(
            X_test=task['y_test'],
            X_pred=X_recon,
            X_generated=X_gen
        )
        scores.append(score)

        print(f"  R2: {score['r2_reconstruction']:.4f}, FID: {score['fid_generation']:.4f}")

    print(f"\n✓ Completed {task_count} task(s)")
    # Don't compute mean of dicts
    return scores


def main():
    """Run all tests"""
    print("\n" + "#" * 60)
    print("# Neural Activity Generation Package Test Suite")
    print("#" * 60 + "\n")

    try:
        # Test environment
        env, task = test_environment()

        # Test agent
        agent, predictions = test_agent(env, task)

        # Test evaluation
        score = test_evaluation(env, task, predictions)

        # Test full workflow
        scores = test_full_workflow()

        print("\n" + "#" * 60)
        print("# ALL TESTS PASSED! ✓")
        print("#" * 60)
        print(f"\nPackage is working correctly!")

    except Exception as e:
        print("\n" + "#" * 60)
        print("# TEST FAILED! ✗")
        print("#" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
