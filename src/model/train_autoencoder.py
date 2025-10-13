#trains autoencoder on benign data, loads data, normalizes features, trains autoencoder, and saves both model and scalar
#siri

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm

from .autoencoder import Autoencoder
from . import config as cfg

def load_data(file_path):
    #loads csv
    df = pd.read_csv(file_path)
    
    #assuming last column is label only use benign 
    if "label" in df.columns:
        df = df[df["label"] == 0] 
        df = df.drop(columns=["label"])

    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    #saving scalar for later
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, cfg.SCALER_SAVE_PATH)
    print(f"scaler saved to {cfg.SCALER_SAVE_PATH}")

    dataset = TensorDataset(X_tensor)
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
        for (x_batch,) in tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS}"):
            x_batch = x_batch.to(cfg.DEVICE)
            optimizer.zero_grad()

            reconstructed = model(x_batch)
            loss = criterion(reconstructed, x_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{cfg.EPOCHS}] | Loss: {avg_loss:.6f}")

    #save
    torch.save(model.state_dict(), cfg.MODEL_SAVE_PATH)
    print(f"[INFO] Model saved to {cfg.MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()

