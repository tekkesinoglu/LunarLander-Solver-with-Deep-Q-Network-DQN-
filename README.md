# 🚀 LunarLander Solver with Deep Q-Network (DQN)

This repository contains a PyTorch implementation of a Deep Q-Network (DQN) agent capable of solving the **LunarLander-v3** environment from Gymnasium. The agent learns to navigate and land the spacecraft safely between the flags using reinforcement learning.

![Lunar Lander](https://gymnasium.farama.org/_images/lunar_lander.gif)

## 🧠 Project Overview

The goal of this project is to train an autonomous agent to control a lunar lander. The agent receives 8 inputs (coordinates, velocity, angle, etc.) and decides on 4 possible actions (do nothing, fire main engine, fire left engine, fire right engine).

### Key Features
* **Deep Q-Network (DQN):** Utilizes a neural network with two hidden layers (64 nodes each) to approximate Q-values.
* **Experience Replay:** Stores past transitions to break correlation between consecutive samples during training.
* **Epsilon-Greedy Strategy:** Balances exploration (trying new things) and exploitation (using learned knowledge).
* **PyTorch Backend:** Efficient training using GPU acceleration if available.

## 🛠️ Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/KULLANICI_ADIN/LunarLander-DQN.git](https://github.com/KULLANICI_ADIN/LunarLander-DQN.git)
    cd LunarLander-DQN
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

### Training the Agent
To train the model from scratch, run:
```bash
python train.py
