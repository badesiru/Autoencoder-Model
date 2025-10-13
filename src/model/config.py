#all central config
import os
import torch

#paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")


TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
#VAL_FILE = os.path.join(DATA_DIR, "val.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")

#this is temp we can change later based on our dataset
INPUT_DIM = 63 # number of features           
LATENT_DIM = 4         
HIDDEN_DIMS = [32, 16]  

#training settings
BATCH_SIZE = 256
LEARNING_RATE = 5e-4
EPOCHS = 30
VALIDATION_SPLIT = 0.2
SEED = 42

#devide setuo
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.xpu.is_available():
    DEVICE = torch.device("xpu")
else:
    DEVICE = torch.device("cpu")

print(f"[INFO] Using device: {DEVICE}")
#outpout
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "autoencoder.pt")
SCALER_SAVE_PATH = os.path.join(MODEL_DIR, "scaler.pkl")