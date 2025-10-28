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
INPUT_DIM = 76 # number of features           
LATENT_DIM = 8
HIDDEN_DIMS = [48,24]

#training settings
#best for mon, wednesday: 1024, 1e-04, 5
#best for all: 1024, 1e-04, 2, 1124,
BATCH_SIZE = 1024
LEARNING_RATE = 1e-04 
EPOCHS = 2 #15-20
VALIDATION_SPLIT = 0.2
SEED = 1124 ##best seed for entire set is 1124
DROPOUT_P = 0.1
NOISE_STD = 0.01
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