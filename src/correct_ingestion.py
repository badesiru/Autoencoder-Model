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
#benign_df = df[df['Label'] == 'BENIGN']
#attack_df = test[df['Label'] != 'BENIGN']

# (2) concat all files and split into train/test
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.columns = df.columns.str.strip() # strip whitespace from column names
benign_df = df[df['Label'] == 'BENIGN']
attack_df = df[df['Label'] != 'BENIGN']


#CICIDS2017 ONLY --> remove "attemped" samples from attack_df
print("Total attack samples before attack:", len(attack_df))
#mask_attempt = attack_df['Label'].str.contains(r'\bAttempt', case=False, na=False) #creates a mask for all rows that contain 'Attempt'
attack_df = attack_df[~attack_df.iloc[:, -1].astype(str).str.contains("Attempted")].reset_index(drop=True)
#attempted_flows = attack_df[mask_attempt]
#attack_df = attack_df[~mask_attempt].reset_index(drop=True) #remove from attack
#attempted_flows['Label'] = 'BENIGN' #relabel as benign
#benign_df = pd.concat([benign_df, attempted_flows], ignore_index=True)

print("Total benign samples:", len(benign_df))
print("Total attack samples:", len(attack_df))


#------------------ SPLIT TEST/TRAIN/VAL ------------------#


#Normalize all the attack labels
attack_df["Label"] = attack_df["Label"].astype(str).str.strip().str.title()

#split attack into val and test
val_attack_list, test_attack_list = [], []
for attack_type, grp in attack_df.groupby("Label"):
    val_part, test_part = train_test_split(
        grp, test_size=0.90, random_state=42, shuffle=True
    )
    val_attack_list.append(val_part)
    test_attack_list.append(test_part)
    print(f"[ATTACK TYPE] {attack_type:<25} → val={len(val_part):6}, test={len(test_part):6}")

# Combine all the validation/test attack samples
val_attack = pd.concat(val_attack_list, ignore_index=True)
test_attack = pd.concat(test_attack_list, ignore_index=True)

#downsample the amount of attacks to rebalance test set

attack_half_list = []
for attack_type, grp in attack_df.groupby("Label"):
    half_sample = grp.sample(frac=0.5, random_state=42)
    attack_half_list.append(half_sample)
    print(f"[DOWNSAMPLE] {attack_type:<30} kept {len(half_sample)} of {len(grp)} attacks")
test_attack = pd.concat(attack_half_list, ignore_index=True)

#Create our train/test/validation splits
train_benign, test_benign = train_test_split(
    benign_df, 
    test_size=0.3, 
    random_state=42, 
    shuffle=True
)
"""
val_benign, test_benign = train_test_split(
    temp_benign,
    test_size=2/3, 
    random_state=42, 
    shuffle=True
)
"""
#assemble the final splits
train = train_benign.copy()
#val = pd.concat([val_benign, val_attack], ignore_index=True)
test = pd.concat([test_benign, test_attack], ignore_index=True)
print(f"[DEBUG] Test set composition → benign={np.sum(test['Label'] == 'BENIGN')}, attacks={np.sum(test['Label'] != 'BENIGN')}")



#remove benign column
train = train.drop(columns=['Label'], errors='ignore')

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
#val = val.drop(columns=[col for col in drop_features if col in val.columns], errors='ignore')

#[NOTE] commented out for selecting top features based on previous analysis
#train = train[train.columns.intersection(top_features)]
#test = test[test.columns.intersection(top_features)]

#Print final training/testing sshape
print(f"Final shape for training list: {train.shape}") 
print(f"Final shape for Testing list: {test.shape}") 

#Create target array for test set 
test["Label"] = test["Label"].apply(lambda x: 1 if x!="BENIGN" else 0) #attack=1, benign=0
#val["Label"] = val["Label"].apply(lambda x: 1 if x!="BENIGN" else 0) #attack=1, benign=0
target = test["Label"].to_numpy() #save target array
#val_target = val["Label"].to_numpy() #save validation target array
#np.save("val_target.npy", val_target)
np.save("target.npy", target) 

#print number of benign and attack samples in test set
print("[TEST SET] Number of benign (0):", np.sum(target == 0))
print("[TEST SET] Number of attack (1):", np.sum(target == 1))
 
#now remove label column from test
test = test.drop(columns=['Label'], errors='ignore')
#val = val.drop(columns=['Label'], errors='ignore')



#Convert to CSV file
output_path = "data/processed/train.csv"
train.to_csv(output_path, index=False)
print(f"Processed data saved to {output_path}")
output_path = "data/processed/test.csv"
test.to_csv(output_path, index=False)
print(f"Processed data saved to {output_path}")
#output_path = "data/processed/val.csv"
#val.to_csv(output_path, index=False)
#print(f"Processed data saved to {output_path}")





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

#----------LOGIC FOR M/W ONLY -------------------#
"""
#Normalize all the attack labels
attack_df["Label"] = attack_df["Label"].astype(str).str.strip().str.title()

#split attack into val and test
val_attack_list, test_attack_list = [], []
for attack_type, grp in attack_df.groupby("Label"):
    val_part, test_part = train_test_split(
        grp, test_size=0.90, random_state=42, shuffle=True
    )
    val_attack_list.append(val_part)
    test_attack_list.append(test_part)
    print(f"[ATTACK TYPE] {attack_type:<25} → val={len(val_part):6}, test={len(test_part):6}")

# Combine all the validation/test attack samples
val_attack = pd.concat(val_attack_list, ignore_index=True)
test_attack = pd.concat(test_attack_list, ignore_index=True)


val_benign, test_benign = train_test_split(
    test_benign_df,
    test_size=0.9, 
    random_state=42, 
    shuffle=True
)

#assemble the final splits
train = benign_df.copy()
val = pd.concat([val_benign, val_attack], ignore_index=True)
test = pd.concat([test_benign, test_attack], ignore_index=True)
print(f"[DEBUG] Test set composition → benign={np.sum(test['Label'] == 'BENIGN')}, attacks={np.sum(test['Label'] != 'BENIGN')}")

"""

