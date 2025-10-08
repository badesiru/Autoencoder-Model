import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

#load csv data into memory
folder = "MachineLearningCVE"

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

#normalize the data
data[numeric_cols] = StandardScaler().fit_transform(data[numeric_cols])

#data correlation
corr_matrix = data[numeric_cols].corr(method='pearson')
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap (Pearson)")
plt.show()
  # or 'spearman'
#output everything into vector 
