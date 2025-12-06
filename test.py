import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import torch.nn as nn
import numpy as np
from train import DQNAgent # DQNAgent sınıfını train.py'den çeker

def test_model():
    # Setup Environment
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    
    # Wrap environment for video recording
    env = RecordVideo(env, video_folder="videos", episode_trigger=lambda e: True)

    # Initialize Agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)
    
    # Load Trained Weights
    try:
        agent.load("lunar_lander_final.pth")
        print("Model loaded successfully.")
    except:
        print("Error: Model file not found. Please train the model first.")
        return

    agent.epsilon = 0.0  # Disable exploration for testing
    
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    print("Starting simulation...")
    
    while not done:
        action = agent.act(state)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    print(f"Simulation Complete. Total Score: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    test_model()
