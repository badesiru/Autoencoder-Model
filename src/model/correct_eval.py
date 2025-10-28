# Evaluates trained autoencoder on test data for performance assessment
# Measures reconstruction quality and anomaly detection capability
# Author: dheeraj
import os
import torch
import numpy as np
import pandas as pd
import joblib
from src.model.autoencoder import Autoencoder
from sklearn.metrics import precision_score, recall_score
import src.model.config as cfg


def evaluate_autoencoder():

    #import and process
    # Load test dataset from processed CSV file
    # DataIngestion.py has already handled scaling and label removal
    df = pd.read_csv(cfg.TEST_FILE)
    df= df.replace([np.inf, -np.inf], np.nan) 
    scaler = joblib.load(cfg.SCALER_SAVE_PATH)
    X_scaled = scaler.transform(df.values)
    # Convert directly to tensor (all preprocessing done by DataIngestion.py)
    X_test = torch.tensor(X_scaled, dtype=torch.float32)
    
    #create target
    target = np.load("target.npy")
    np.savetxt("targets.csv", target, delimiter=",", fmt="%.3f")

    print("Mean of features:", df.mean().mean())
    print("Std of features:", df.std().mean())

    if not os.path.exists(cfg.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model not found at {cfg.MODEL_SAVE_PATH}")
    if not os.path.exists(cfg.TEST_FILE):
        raise FileNotFoundError(f"Test data not found at {cfg.TEST_FILE}")
    # Load the trained autoencoder model
    # Create model with same architecture as training
    model = Autoencoder(cfg.INPUT_DIM, cfg.LATENT_DIM, cfg.HIDDEN_DIMS)
    # Load the trained weights from saved model file
    model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH, map_location=cfg.DEVICE))
    # Move model to appropriate device (CPU/GPU)
    model.to(cfg.DEVICE)
    # Set model to evaluation mode (disables dropout, batch norm updates)
    model.eval()
    # Load and preprocess test data

    with torch.no_grad():
        X_test = X_test.to(cfg.DEVICE)
        X_scaled_test_recon = model(X_test)
    loss_test = torch.mean((X_scaled_test_recon - X_test) ** 2, dim=1).cpu().numpy()
    min_loss, max_loss = loss_test.min(), loss_test.max()
    step = max_loss / (min_loss*10+1e-10)
    thresh = min_loss
    while thresh <= max_loss/0.9:
        thresh*=1.1
        # select a threshold for predictions
        predictions = (loss_test > thresh).astype(np.int64)
        if predictions.sum() == 0:
            continue
        # Calculate precision and recall
        precision = precision_score(target, predictions )
        recall = recall_score(target, predictions )
        if(precision>=0.7 and recall>=0.7):
            print("Thresh : ", thresh)
            print("Total anomalies : ", target.sum())
            print("Detected anomalies : ", predictions.sum())
            print("Correct anomalies : ", (predictions * target).sum())
            print("Missed anomalies : ", ((1 - predictions) * target).sum())
            f1 = (2 * precision * recall) / (precision + recall + 1e-10)  # avoid divide-by-zero

            # Compute confusion matrix components
            TP = (predictions * target).sum()
            FP = (predictions * (1 - target)).sum()
            FN = ((1 - predictions) * target).sum()
            TN = ((1 - predictions) * (1 - target)).sum()

            # Compute rates
            FPR = FP / (FP + TN + 1e-10)
            FNR = FN / (FN + TP + 1e-10)

            # Print all results
            print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
            print(f"FPR: {FPR:.3f}, FNR: {FNR:.3f}, FN: {FN}, FP: {FP}, TP: {TP}, TN: {TN}")

if __name__ == "__main__":
    evaluate_autoencoder()
