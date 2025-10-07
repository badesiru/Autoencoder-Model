#all central config


import os

#paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")


TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
VAL_FILE = os.path.join(DATA_DIR, "val.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")

#this is temp we can change later based on our dataset
INPUT_DIM = 78             
LATENT_DIM = 16            
HIDDEN_DIMS = [64, 32, 16]  

#training settings
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 50
VALIDATION_SPLIT = 0.2
SEED = 42

#devide setuo
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#outpout
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "autoencoder.pt")
SCALER_SAVE_PATH = os.path.join(MODEL_DIR, "scaler.pkl")



