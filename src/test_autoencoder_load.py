#makeing sure its saved corely and test

import torch
import numpy as np
import joblib
import os
from src.model.autoencoder import Autoencoder
import src.model.config as cfg


def test_autoencoder():
    print("Autoencoder load test")

    #check paths
    if not os.path.exists(cfg.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"model not found at {cfg.MODEL_SAVE_PATH}")
    if not os.path.exists(cfg.SCALER_SAVE_PATH):
        raise FileNotFoundError(f"scaler not found at {cfg.SCALER_SAVE_PATH}")
    print("mdel and scaler files found.")

    #load both 
    model = Autoencoder(cfg.INPUT_DIM, cfg.LATENT_DIM, cfg.HIDDEN_DIMS)
    model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH, map_location="cpu"))
    model.eval()
    print("model loaded successfully.")

    scaler = joblib.load(cfg.SCALER_SAVE_PATH)
    print("scaler loaded successfully.")

    #test input - one sample same as input features
    sample = np.random.rand(1, cfg.INPUT_DIM) 
    sample_scaled = scaler.transform(sample)
    sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32)

    #pass through model 
    with torch.no_grad():
        reconstructed = model(sample_tensor)
        mse = torch.mean((sample_tensor - reconstructed) ** 2).item()

    print(f"reconstruction error: {mse:.6f}")

    #sanity check 
    if mse < 0.1:
        print("autoencoder reconstruction - low error")
    else:
        print("reconstruction error  high")


if __name__ == "__main__":
    test_autoencoder()
