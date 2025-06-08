# BESS Control with Deep Q-Learning

This project implements a Battery Energy Storage System (BESS) controller using Deep Reinforcement Learning, specifically Double Deep Q-Network (Double DQN) algorithms. The system is designed to optimize battery operation in a microgrid environment with solar generation, variable loads, and time-of-use electricity pricing.

## Project Structure

```
├── BESS_Control_With_Deep_Q-Learning.ipynb   # Main Jupyter notebook for running experiments
├── double_dqn_agent.py                       # Implementation of the Double DQN agent
├── double_dqn_model.py                       # Neural network architecture for the Double DQN
├── load_profile_data_loader.py               # Utility for loading and preprocessing load data
├── load_profile_data_nov2024.csv             # Sample load profile data
├── microgrid_gym_env.py                      # Gymnasium environment for the microgrid simulation
├── requirements.txt                          # Python dependencies
├── model_checkpoints/                        # Saved model weights
│   └── double_dqn_model_weights.pth          # Best trained model weights
└── runs/                                     # TensorBoard log directories
    └── ...                                   # Training session logs
```

## Setup Instructions

### Prerequisites

- Python 3.11
- Git (optional, for cloning the repository)

### Setting Up the Environment

1. **Install Python 3.11**:
   
   For macOS (using Homebrew):
   ```bash
   brew install python@3.11
   ```
   
   For other operating systems, download from [python.org](https://www.python.org/downloads/).

2. **Create and activate a virtual environment**:
   ```bash
   # Navigate to the project directory
   cd "/Users/your-username/path-to/deep_q-learning"
   
   # Create a virtual environment
   python3.11 -m venv venv
   
   # Activate the virtual environment
   source venv/bin/activate
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

### Using TensorBoard

1. **Start TensorBoard**:
   ```bash
   # Ensure you're in the project directory with the virtual environment activated
   tensorboard --logdir=runs
   ```

2. **Access TensorBoard in your browser**:
   
   Open a web browser and navigate to:
   ```
   http://localhost:6006
   ```
   
   This will show you training metrics, reward curves, and other visualization data for your training runs.

### Running the Jupyter Notebook

1. **Install the virtual environment as a Jupyter kernel**:
   ```bash
   pip install ipykernel
   python -m ipykernel install --user --name=bess-dqn-venv --display-name="BESS DQN (Python 3.11)"
   ```

2. **Open the notebook in VS Code**:
   - Launch VS Code
   - Open the project folder
   - Open the `BESS_Control_With_Deep_Q-Learning.ipynb` file
   - Select the `BESS DQN (Python 3.11)` kernel from the kernel picker in the top-right corner of the notebook

3. **Run the notebook**:
   - You can run cells individually by clicking the play button next to each cell, or
   - Run all cells using the "Run All" button at the top of the notebook

## Project Components

### microgrid_gym_env.py

This file implements a Gymnasium environment that simulates a microgrid with solar generation, battery storage, and grid connection. It includes:

- `EnvState`: A dataclass that holds the state of the environment at each timestep
- `MicrogridEnv`: The main environment class that implements the Gymnasium interface:
  - `reset()`: Resets the environment to the initial state
  - `step(action)`: Advances the simulation by one step, taking an action and returning the next state, reward, etc.
  - Various helper methods for calculating rewards, state transitions, etc.

### double_dqn_model.py

This file implements the neural network architecture for the Double DQN algorithm:

- `DoubleDQNModel`: A PyTorch neural network with:
  - Value network for state value estimation
  - Advantage network with noisy linear layers for exploration
  - Dueling architecture that separates value and advantage estimation

### double_dqn_agent.py

This file implements the reinforcement learning agent using the Double DQN algorithm:

- `DoubleDQNAgent`: The main agent class that handles:
  - Experience collection and replay
  - Training the neural network
  - Target network synchronization
  - Exploration strategy
  - TensorBoard logging

### load_profile_data_loader.py

This file contains the `LoadProfileDataLoader` class for:

- Loading load profile data from CSV files
- Preprocessing the data (normalization, feature engineering)
- Converting data into formats suitable for the environment

### BESS_Control_With_Deep_Q-Learning.ipynb

This Jupyter notebook serves as the main interface for the project, allowing you to:

- Load and preprocess data
- Set up the environment and agent
- Train the agent
- Evaluate its performance
- Visualize results
- Compare the DQN agent against a baseline rule-based controller

## Additional Information

### Training Tips

- The project uses TensorBoard for monitoring training progress
- Training can take several hours depending on your hardware
- Pre-trained models are saved in the `model_checkpoints` directory
- You can adjust hyperparameters in the notebook for different training configurations

### Troubleshooting

- If you encounter CUDA errors, try setting `device='cpu'` in the notebook
- If the notebook fails to recognize the environment, ensure you've activated the correct virtual environment
- For visualization issues, ensure you have the correct matplotlib version installed

## References

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [TensorBoard Documentation](https://www.tensorflow.org/tensorboard)
- [PTAN (PyTorch Agent Net) Documentation](https://github.com/Shmuma/ptan)