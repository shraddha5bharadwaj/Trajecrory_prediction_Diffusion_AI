# Trajecrory_prediction_Diffusion_AI

# Diffusion-Based Cursor Trajectory Generation

This project implements a **Denoising Diffusion Probabilistic Model (DDPM)** to generate synthetic human-like cursor trajectories. By modeling movement as a probabilistic denoising process, the system can "dream" realistic future paths based on historical movement data.

---

## 🚀 Project Overview

The core objective is to predict the next cursor coordinates ($x, y$) by conditioning a diffusion model on a window of past trajectory points. This approach captures the stochastic nature of human motor control more effectively than traditional deterministic regression.

### 1. Data Pipeline
* **Source:** Loads cursor data from `matched_pos.csv`.
* **Preprocessing:** * Handles missing values via forward-filling (`ffill`).
    * Groups data by `trial_type` to preserve movement continuity.
    * Generates sequences with a fixed **window length of 50 steps**.
* **Dataset:** Implements a custom `FixedHistoryTrajectoryDataset` to feed history-target pairs into the model.

### 2. Model Architecture
The architecture utilizes a conditioned denoising approach:
* **DenoisingTrajectoryModel:** * **History Encoder:** A 1D Convolutional network with Adaptive Average Pooling to extract spatial-temporal features from the past 50 points.
    * **Noisy Position Projector:** Embeds the current noisy state into the latent space.
    * **Time Embedding:** Uses Sinusoidal Position Embeddings to provide the model with awareness of the current diffusion timestep ($t$).
    * **Output Head:** An MLP that concatenates history, time, and noisy input to predict the added Gaussian noise.

### 3. Diffusion Process
Implemented via the `StepwiseGaussianDiffusion` class:
* **Forward Process:** Adds Gaussian noise to the target position using a linear beta schedule ($1e^{-4}$ to $0.02$) over 1000 timesteps.
* **Reverse Process (Sampling):** Iteratively removes noise to recover the clean target position, conditioned on the movement history.

### 4. Training Details
* **Loss Function:** Mean Squared Error (MSE) between predicted and actual noise.
* **Optimizer:** Adam ($lr=1e^{-4}$).
* **Regularization:** Includes an **Early Stopping** mechanism that monitors validation loss to prevent overfitting.
* **Storage:** Saves the optimal weights to `best_model.pt`.

---

## 📊 Visualization & Results
The notebook includes specialized plotting tools to evaluate performance:
* **Trajectory Quiver Plots:** Uses `matplotlib`'s quiver function with a `plasma` colormap to show the direction and velocity of generated paths.
* **Ground Truth Comparison:** Overlays generated trajectories against real "True Future" data to verify spatial accuracy and movement fluidity.

---

## 🛠 Dependencies
* Python 3.11+
* PyTorch
* Pandas & NumPy
* Matplotlib
* Scikit-learn
<img width="523" height="409" alt="Screenshot 2026-04-08 at 7 27 37 PM" src="https://github.com/user-attachments/assets/468cec57-c590-4bbb-948b-31deebce7b91" />
