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
    loss_test = ((X_test - X_recon)**2).mean(dim=1).cpu().numpy()


    
    #find best thresh
    df_eval = pd.DataFrame({'error': loss_test, 'y_true': target})
    results = thresholdTuning(df_eval, iterations=100, start=0, end=0.99)
    best_idx = results['f1'].idxmax()
    best_threshold = results.loc[best_idx, 'threshold']
    best_f1 = results.loc[best_idx, 'f1']
    
    
    #fine tune thresholds
    thresholds = np.linspace(best_threshold * 0.5, best_threshold *20, 400)
    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        preds = (loss_test > t).astype(int)
        prec = precision_score(target, preds, pos_label=1, zero_division=0)
        rec  = recall_score(target, preds, pos_label=1, zero_division=0)
        f1   = f1_score(target, preds, pos_label=1, zero_division=0)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)


    #Choose metric to maximize

    # === Option A: maximize F1 (current behavior)
    best_idx = np.argmax(f1s)
    best_threshold = thresholds[best_idx]
    print(f"\n[F1-opt] Threshold={best_threshold:.8f}, "
    f"P={precisions[best_idx]:.3f}, R={recalls[best_idx]:.3f}, F1={f1s[best_idx]:.3f}")

    # === Option B: choose threshold that achieves desired precision target
    target_precision = 0.95  # <-- change to your goal
    idxs = np.where(np.array(precisions) >= target_precision)[0]
    if len(idxs):
        p_target_idx = idxs[0]  # first threshold that meets target precision
        print(f"\n[P≥{target_precision}] Threshold={thresholds[p_target_idx]:.8f}, "
            f"P={precisions[p_target_idx]:.3f}, R={recalls[p_target_idx]:.3f}, F1={f1s[p_target_idx]:.3f}")
    else:
        print(f"\nNo threshold reached precision {target_precision}")

    # === Option C (similarly for recall target)
    target_recall = 0.95
    idxs = np.where(np.array(recalls) >= target_recall)[0]
    if len(idxs):
        r_target_idx = idxs[-1]  # last threshold before recall drops below target
        print(f"\n[R≥{target_recall}] Threshold={thresholds[r_target_idx]:.8f}, "
            f"P={precisions[r_target_idx]:.3f}, R={recalls[r_target_idx]:.3f}, F1={f1s[r_target_idx]:.3f}")
        


    #display best threshold results
    predictions = (loss_test > best_threshold).astype(np.int64)
    precision = precision_score(target, predictions )
    recall = recall_score(target, predictions)
    print("Total anomalies : ", target.sum())
    print(f"\nBest threshold: {best_threshold:.9f} (F1 = {best_f1:.3f})")
    print("Detected anomalies : ", predictions.sum())
    print("Correct anomalies : ", (predictions*target).sum())
    print("Missed anomalies : ", ((1-predictions)*target).sum())
    print(f'Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {(2*precision*recall)/(precision + recall):.2f}')
  


if __name__ == "__main__":
    # Run the evaluation when script is executed directly
    evaluate_autoencoder()
