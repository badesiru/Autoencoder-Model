#feedforward autoencoder using pyrotch, rectructs benign network traffic
#siri

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
import joblib
from tqdm import tqdm

from .autoencoder import Autoencoder
from . import config as cfg

def kl_divergence(p,p_hat):
    ##return divergence from target sparsity and average activation
    return torch.sum(
        p * torch.log(p / (p_hat + 1e-10)) +
        (1 - p) * torch.log((1 - p) / (1 - p_hat + 1e-10))
    )
def load_data(file_path):
    #loads csv
    df = pd.read_csv(file_path)
    print(df.head(1))
    print(df.dtypes)
    #assuming last column is label only use benign 

    #create scaler,  Options: StandardScaler, MinMaxScaler,  MaxAbsScaler
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df.values)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    #saving scalar for later
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, cfg.SCALER_SAVE_PATH)
    print(f"scaler saved to {cfg.SCALER_SAVE_PATH}")

    dataset = TensorDataset(X_tensor, X_tensor)
    return dataset


def train():
    torch.manual_seed(cfg.SEED)

    #load data
    dataset = load_data(cfg.TRAIN_FILE)
    dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)

    #initialize model 
    model = Autoencoder(cfg.INPUT_DIM, cfg.LATENT_DIM, cfg.HIDDEN_DIMS).to(cfg.DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    #training loop 
    model.train()
    for epoch in range(cfg.EPOCHS):
        epoch_loss = 0
        # y_batch is target, target is same as train input for autoencoder
        for x_batch, y_batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS}"):
            x_batch = x_batch.to(cfg.DEVICE)
            
            # === Add Gaussian noise to input ===
            noisy_input = x_batch + torch.randn_like(x_batch) * cfg.NOISE_STD

            optimizer.zero_grad()
            reconstructed = model(noisy_input)
            recon_loss = criterion(reconstructed, x_batch)  # target is clean

             # === Sparsity Regularization ===
            p_hat = torch.mean(model.latent, dim=0)  # average activation per latent neuron
            sparsity_loss = kl_divergence(torch.full_like(p_hat, cfg.SPARSITY_TARGET), p_hat)
            total_loss = recon_loss + cfg.SPARSITY_WEIGHT * sparsity_loss
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
        avg_activation = torch.mean(model.latent).item()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{cfg.EPOCHS}] | Loss: {avg_loss:.6f} | Avg latent activation: {avg_activation:.4f}")

    #save
    torch.save(model.state_dict(), cfg.MODEL_SAVE_PATH)
    print(f"[INFO] Model saved to {cfg.MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()
