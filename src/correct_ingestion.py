#Correct ingestion mechanism for training dataset
#Author: Sully Mrkva

import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#load csv data into memory
files = [
    "src/MachineLearningCVE/Wednesday-WorkingHours.csv",
    "src/MachineLearningCVE/Friday-WorkingHours.csv",
    "src/MachineLearningCVE/Tuesday-WorkingHours.csv", 
    "src/MachineLearningCVE/Thursday-WorkingHours.csv",
    "src/MachineLearningCVE/Monday-WorkingHours.csv"
]

# (1) Read training file into data [Monday train, Wednesday test]
#df = pd.read_csv("src/MachineLearningCVE/Monday-WorkingHours.csv")
#df.columns = df.columns.str.strip() # strip whitespace from column names
#test = pd.read_csv("src/MachineLearningCVE/Wednesday-WorkingHours.csv")
#test.columns = test.columns.str.strip() # strip whitespace from column names

# (2) concat all files and split into train/test

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.columns = df.columns.str.strip() # strip whitespace from column names
benign_df = df[df['Label'] == 'BENIGN']
attack_df = df[df['Label'] != 'BENIGN']

print("Total benign samples:", len(benign_df))
print("Total attack samples:", len(attack_df))
train_benign, test_benign = train_test_split(
    benign_df, 
    test_size=0.2, 
    random_state=42, 
    shuffle=True
)
train = train_benign.copy()
test  = pd.concat([test_benign, attack_df], ignore_index=True)
"""
"""

#remove benign column
train = df.drop(columns=['Label'], errors='ignore')

#some data cleaning
train= train.replace([np.inf, -np.inf], np.nan) #replace infinite values with NaN
train = train.dropna() #Drop rows with zero values in numeric columns
test= test.replace([np.inf, -np.inf], np.nan) #replace infinite values with NaN
test = test.dropna() #Drop rows with zero values in numeric columns

#Feature removal (categorical,identifier columns)
drop_features = [
    'Flow ID',
    'Src IP',
    'Src Port',
    'Dst IP',
    'Dst Port',
    'Protocol',
    'Timestamp',

]

#drop features
train = train.drop(columns=[col for col in drop_features if col in train.columns], errors='ignore')
test = test.drop(columns=[col for col in drop_features if col in test.columns], errors='ignore')

#[NOTE] commented out for selecting top features based on previous analysis
#train = train[train.columns.intersection(top_features)]
#test = test[test.columns.intersection(top_features)]

#Print final training/testing sshape
print(f"Final shape for training list: {train.shape}") 
print(f"Final shape for Testing list: {test.shape}") 

#Create target array for test set 
test["Label"] = test["Label"].apply(lambda x: 1 if x!="BENIGN" else 0) #attack=1, benign=0
target = test["Label"].to_numpy() #save target array
np.save("target.npy", target) 

#print number of benign and attack samples in test set
print("Number of benign (0):", np.sum(target == 0))
print("Number of attack (1):", np.sum(target == 1))
 
#now remove label column from test
test = test.drop(columns=['Label'], errors='ignore')


#Convert to CSV file
output_path = "data/processed/train.csv"
train.to_csv(output_path, index=False)
print(f"Processed data saved to {output_path}")
output_path = "data/processed/test.csv"
test.to_csv(output_path, index=False)
print(f"Processed data saved to {output_path}")

"""#top features based on importance from previous analysis
top_features = [
    'Bwd Packet Length Std',
    'Idle Mean',
    'Avg Bwd Segment Size',
    'act_data_pkt_fwd',
    'PSH Flag Count',
    'Idle Max',
    'Subflow Fwd Bytes',
    'Bwd Packet Length Mean',
    'Average Packet Size',
    'Bwd Header Length',
    'Total Length of Fwd Packets',
    'Subflow Bwd Packets',
    'Bwd Packets/s',
    'Total Fwd Packets',
    'Fwd Packet Length Min',
    'Subflow Bwd Bytes',
    'Total Backward Packets',
    'Max Packet Length',
    'Total Length of Bwd Packets',
    'Bwd Packet Length Min', 'Label'
]
drop_features = [
    # --- Fwd/Bwd Length redundancy ---
    'Avg Fwd Segment Size',
    'Avg Bwd Segment Size',

    # --- Subflow duplicates ---
    'Subflow Fwd Packets',
    'Subflow Bwd Packets',
    'Subflow Fwd Bytes',
    'Subflow Bwd Bytes',

    # --- Totals overlapping with Subflow ---
    'Total Length of Fwd Packets',
    'Total Length of Bwd Packets',

    # --- IAT / Duration redundancy ---
    'Fwd IAT Total',
    'Fwd IAT Max',
    'Flow IAT Max',

    # --- Idle/Active duplicates ---
    'Idle Max',
    'Idle Min',
    'Active Max',
    'Active Std',

    #desination port
    'Destination Port'
]
"""