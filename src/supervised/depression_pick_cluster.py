import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

# PASUL 1: încarcă datele
# datele tale curățate
data_cleaned = pd.read_csv('data/cleaned/cleaned_data.csv')
clusters = pd.read_csv('data/interpreted/interpreted_data.csv')

# Separăm clusterul de restul datelor
cluster_centers = clusters.drop(columns=['Cluster'])

# PASUL 2: Atribuim fiecare instanță la cel mai apropiat cluster
# Calculăm distanța de fiecare centru
distances = cdist(data_cleaned.values, cluster_centers.values)

# Pentru fiecare instanță alegem clusterul cu distanța cea mai mică
assigned_clusters = np.argmin(distances, axis=1)

# PASUL 3: Adăugăm coloana "Depresie"
# clusterul 3 => depresie 1, altfel 0
data_cleaned['Cluster'] = assigned_clusters
data_cleaned['Depresie'] = ((assigned_clusters == 3) | (assigned_clusters == 0)).astype(int)

# PASUL 4: Salvăm rezultatul
data_cleaned.to_csv('data/label_depresie/data_cleaned_with_depresie.csv', index=False)

print(data_cleaned.head())
