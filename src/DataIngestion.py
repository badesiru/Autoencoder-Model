import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#load csv data into memory
folder = "src\\MachineLearningCVE"

#combine all the daily datasets
dfs = [pd.read_csv(os.path.join(folder, f)) for f in os.listdir(folder) if f.endswith(".csv")]
data = pd.concat(dfs, ignore_index=True)
data.columns = data.columns.str.strip() # strip whitespace from column names
#let's view all unique attack types
#print(data['Label'].unique()) 

#Separate numeric and categorical columns
numeric_cols = data.select_dtypes(include=["number"]).columns
#MLReady dataset is being used, so no need to

#replace infinite values with NaN
data[numeric_cols] = data[numeric_cols].replace([np.inf, -np.inf], np.nan)

#Drop rows with zero values in numeric columns
data = data.dropna(subset=numeric_cols) 

#normalize the data [Commented out because of scaling in training script]
#data[numeric_cols] = StandardScaler().fit_transform(data[numeric_cols])

#data correlation
corr_matrix = data[numeric_cols].corr(method='pearson')
#plt.figure(figsize=(14, 10))
#sns.heatmap(corr_matrix, cmap='coolwarm', linewidths=0.5)
#plt.title("Feature Correlation Heatmap (Pearson)")
#plt.show()
#corr_matrix.to_csv("correlations.csv")

#identify redundant features
threshold = 0.9
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
highly_correlated = [
    column for column in upper.columns if any(upper[column].abs() > threshold)
]
#print(f"\n[{len(highly_correlated)}] Highly correlated features identified:")
#print(highly_correlated)

#drop reduntant features
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
    'Active Std'
]
data_reduced = data.drop(columns=[col for col in drop_features if col in data.columns], errors='ignore')
#print(f"Removed {len(drop_features)} redundant columns. Final shape: {data_reduced.shape}")


#let us make the training set.
benign_data = data_reduced[data_reduced['Label'] == 'BENIGN']
attack_data = data_reduced[data_reduced['Label'] != 'BENIGN'].copy()

#split benign into 80/20 train/test split
X_train, X_benign_test = train_test_split(
    benign_data, test_size=0.2, random_state=42, shuffle=True
)

#concatenate test benign and attack data
test_data = pd.concat([X_benign_test, attack_data], ignore_index=True)

#find nunber of features
X_train = X_train.drop(columns=['Label'], errors='ignore')
true_data = test_data
test_data = test_data.drop(columns=['Label'], errors='ignore')
num_features = X_train.shape[1]
print(num_features)


# Step 4: save files
X_train.to_csv("train.csv", index=False)
print("Number of columns:", X_train.shape[1])
print("Column names:", X_train.columns.tolist())
test_data.to_csv("test.csv", index=False)
true_data.to_csv("true_data.csv", index=False) 
print("Number of columns:", true_data.shape[1])