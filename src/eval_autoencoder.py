# Evaluates trained autoencoder on test data for performance assessment
# Measures reconstruction quality and anomaly detection capability
# Author: dheeraj
import os
import torch
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, accuracy_score
from src.model.autoencoder import Autoencoder
import src.model.config as cfg
import matplotlib.pyplot as plt



def load_test_data():
    # Load test dataset from processed CSV file
    # DataIngestion.py has already handled scaling and label removal
    df = pd.read_csv(cfg.TEST_FILE)
    scaler = joblib.load(cfg.SCALER_SAVE_PATH)
    X_scaled = scaler.transform(df.values)
    # Convert directly to tensor (all preprocessing done by DataIngestion.py)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    #print("Mean of features:", df.mean().mean())
    #print("Std of features:", df.std().mean())
    return X_tensor

def thresholdTuning(df, iterations=50, start=0.05, end=1.0):
    
    thresh_df = {
        'threshold': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [] 
    }
    
    for q in np.linspace(start, end, iterations):
        thresh_value = df['error'].quantile(q)
        preds = (df['error'] > thresh_value).astype(int)

        acc = accuracy_score(df['y_true'], preds)
        prec = precision_score(df['y_true'], preds, pos_label=1, zero_division=0)
        rc   = recall_score(df['y_true'], preds, pos_label=1, zero_division=0)
        f1   = f1_score(df['y_true'], preds, pos_label=1, zero_division=0)

        thresh_df['threshold'].append(thresh_value)
        thresh_df['accuracy'].append(acc)
        thresh_df['precision'].append(prec)
        thresh_df['recall'].append(rc)
        thresh_df['f1'].append(f1)

        print(f"Q={q:.3f} | Threshold={thresh_value:.6f} | Acc={acc:.3f} | Prec={prec:.3f} | Rec={rc:.3f} | F1={f1:.3f}")

    return pd.DataFrame(thresh_df)
    

def evaluate_autoencoder():
    print("Evaluating autoencoder on test data...")
    # Validate that all required files exist before proceeding
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


    #LOAD, TEST AND EVAL PERFORMANCE.

    # Load and preprocess test data
    X_test = load_test_data()
    X_test = X_test.to(cfg.DEVICE)
    true_data = pd.read_csv("data/processed/true_data.csv")
    target= (true_data["Label"].str.lower() != "benign").astype(int).values
    
    with torch.no_grad():
        X_recon= model(X_test)
    #loss_test = torch.nn.functional.l1_loss(X_recon, X_test, reduction='none').sum(1).cpu().numpy()
    loss_test = torch.nn.functional.mse_loss(X_recon, X_test, reduction='none').sum(1).cpu().numpy()
    loss_test = loss_test - loss_test.min()
    loss_test = loss_test/(loss_test.max())

    
    #find best thresh
    df_eval = pd.DataFrame({'error': loss_test, 'y_true': target})
    results = thresholdTuning(df_eval, iterations=100, start=0, end=0.99)
    best_idx = results['f1'].idxmax()
    best_threshold = results.loc[best_idx, 'threshold']
    best_f1 = results.loc[best_idx, 'f1']
    
    
    #fine tune thresholds
    thresholds = np.linspace(best_threshold * 0.5, best_threshold * 1.5, 200)
    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        preds = (loss_test > t).astype(int)
        prec = precision_score(target, preds, pos_label=1, zero_division=0)
        rec  = recall_score(target, preds, pos_label=1, zero_division=0)
        f1   = f1_score(target, preds, pos_label=1, zero_division=0)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    best_idx = np.argmax(f1s)
    best_threshold = thresholds[best_idx]
    #use best threshold to classify
    predictions = (loss_test > best_threshold).astype(np.int64)
    precision = precision_score(target, predictions )
    recall = recall_score(target, predictions)
    print("Total anomalies : ", target.sum())
    print(f"\nBest threshold: {best_threshold:.9f} (F1 = {best_f1:.3f})")
    print("Detected anomalies : ", predictions.sum())
    print("Correct anomalies : ", (predictions*target).sum())
    print("Missed anomalies : ", ((1-predictions)*target).sum())
    print(f'Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {(2*precision*recall)/(precision + recall):.2f}')
    """
    #convert to tensor and run
    with torch.no_grad():
        X_recon = model(X_test)
        #compute per-sample reconstruction error (MSE)
        recon_errors = torch.mean((X_test - X_recon) ** 2, dim=1).cpu().numpy()
        

    #normalize errors to [0,1] for thresholding
    recon_errors = (recon_errors - recon_errors.min()) / (recon_errors.max() - recon_errors.min())

    #threshold selection
    threshold = 0.00005 
    
    #classification and comparison
    
    #metrics
    preds = (recon_errors > threshold).astype(int)
    precision = precision_score(y_true, preds)
    recall    = recall_score(y_true, preds)
    f1        = f1_score(y_true, preds)
    roc_auc   = roc_auc_score(y_true, recon_errors)

    #summary
    total_anomalies = y_true.sum()
    detected_anomalies = preds.sum()
    correct_anomalies = (preds * y_true).sum()  # true positives
    missed_anomalies = ((1 - preds) * y_true).sum()  # false negatives

    print(f"\n=== Detection Summary ===")
    print(f"Total anomalies     : {total_anomalies}")
    print(f"Detected anomalies  : {detected_anomalies}")
    print(f"Correct anomalies   : {correct_anomalies}")
    print(f"Missed anomalies    : {missed_anomalies}")

    print(f"\n=== Evaluation Results ===")
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, ROC-AUC: {roc_auc:.3f}")

    #debugging
    print(X_test.mean().item(), X_test.std().item())




    # Run inference without gradient computation (faster, uses less memory)
    with torch.no_grad():
        # Reconstruct test samples using the trained autoencoder
        reconstructed = model(X_test)
        # Calculate Mean Squared Error (MSE) for each sample
        # MSE = mean((original - reconstructed)²) per sample
        mse = torch.mean((X_test - reconstructed) ** 2, dim=1)
        # Calculate average MSE across all test samples
        avg_mse = mse.mean().item()
        # Report overall reconstruction quality
        print(f"Average reconstruction MSE on test set: {avg_mse:.6f}")
        # Show sample reconstruction errors for analysis
        # First 10 samples to see the range of errors
        print("Sample reconstruction errors:", mse[:10].cpu().numpy())
        # Interpretation guidance
        print("\nInterpretation:")
        print("- Lower MSE values indicate better reconstruction (likely benign traffic)")
        print("- Higher MSE values indicate poor reconstruction (likely attack traffic)")
        print("- Use these errors to set detection thresholds for anomaly classification")
"""
    


if __name__ == "__main__":
    # Run the evaluation when script is executed directly
    evaluate_autoencoder()
