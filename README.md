# Debugging Neural Networks
Numerical experiments for better understanding neural networks.

Goal: Bridge the gap between practical behavior and theoretical models - small autoencoder example + detailed logging.

Installation:
`pip install -r requirements.txt`

Configuration:
- Modify `experiments/config.yaml` according to your needs (input_dim, hidden_dims, lr, epochs, etc.).

Execution:
`python src/train.py`

Results:
A timestamped subdirectory will be created in the runs/ folder containing step_000000.npz files.
These files include:
- param__<name> : weights / biases
- grad__<name> : gradients
- act__<name> : activations
- loss : loss at the time of saving

Analysis:
`python src/analysis.py`  # edit it, specify the run_dir and parameter names
