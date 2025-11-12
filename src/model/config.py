#all central config
import os
import torch

#paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")


TRAIN_FILE = os.path.join(DATA_DIR, "Miraitrain.csv")
#VAL_FILE = os.path.join(DATA_DIR, "val.csv")
TEST_FILE = os.path.join(DATA_DIR, "Miraitest.csv")

INPUT_DIM = 115 # number of features           
LATENT_DIM = 8
HIDDEN_DIMS = [64,36]

BATCH_SIZE = 1024
LEARNING_RATE = 1e-04
EPOCHS = 8 #15-20
VALIDATION_SPLIT = 0.2
SEED = 1124 ##best seed for entire set is 1124
SPARSITY_TARGET = 0.05  # desired average activation per neuron
SPARSITY_WEIGHT = 1e-3  # tuning parameter — start small
DROPOUT_P = 0.1
NOISE_STD = 0.1
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