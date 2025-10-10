# evals trained autoencoder on test data for performance
# dheeraj

import os
import torch
import numpy as np
import pandas as pd
import joblib
from src.model.autoencoder import Autoencoder
import src.model.config as cfg


def load_test_data():
    df = pd.read_csv(cfg.TEST_FILE)
    # If label column exists, drop it for evaluation
    if "label" in df.columns:
        df = df.drop(columns=["label"])
    scaler = joblib.load(cfg.SCALER_SAVE_PATH)
    X = scaler.transform(df.values)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    return X_tensor


def evaluate_autoencoder():
    print("Evaluating autoencoder on test data...")
    if not os.path.exists(cfg.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model not found at {cfg.MODEL_SAVE_PATH}")
    if not os.path.exists(cfg.SCALER_SAVE_PATH):
        raise FileNotFoundError(f"Scaler not found at {cfg.SCALER_SAVE_PATH}")
    if not os.path.exists(cfg.TEST_FILE):
        raise FileNotFoundError(f"Test data not found at {cfg.TEST_FILE}")

    # Load model
    model = Autoencoder(cfg.INPUT_DIM, cfg.LATENT_DIM, cfg.HIDDEN_DIMS)
    model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH, map_location=cfg.DEVICE))
    model.to(cfg.DEVICE)
    model.eval()

    # Load test data
    X_test = load_test_data()
    X_test = X_test.to(cfg.DEVICE)

    # Run inference
    with torch.no_grad():
        reconstructed = model(X_test)
        mse = torch.mean((X_test - reconstructed) ** 2, dim=1)
        avg_mse = mse.mean().item()
        print(f"Average reconstruction MSE on test set: {avg_mse:.6f}")
        # Optionally, print some sample errors
        print("Sample reconstruction errors:", mse[:10].cpu().numpy())

if __name__ == "__main__":
    evaluate_autoencoder()
