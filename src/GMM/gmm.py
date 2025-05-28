import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import seaborn as sns

# =============================
# 1. Încărcare și eșantionare
# =============================
df = pd.read_csv("data/cleaned/cleaned_data.csv")
df_sample = df.sample(n=20000, random_state=42).reset_index(drop=True)

# =============================
# 2. Selectare variabile
# =============================
selected_features = [
    "Age", "Marital Status", "Education Level", "Number of Children",
    "Smoking Status", "Physical Activity Level", "Employment Status", "Income",
    "Alcohol Consumption", "Dietary Habits", "Sleep Patterns",
    "History of Mental Illness", "History of Substance Abuse",
    "Family History of Depression", "Chronic Medical Conditions"
]
X = df_sample[selected_features]

# =============================
# 4. Eliminare variabile cu varianță mică
# =============================
vt = VarianceThreshold(threshold=0.01)
X_var_filtered = vt.fit_transform(X)



# =============================
# 6. GMM clustering pentru interval extins k
# =============================
bic_scores, aic_scores, silhouette_scores = [], [], []
n_range = range(2, 16)

for k in n_range:
    gmm = GaussianMixture(n_components=k, random_state=42)
    labels = gmm.fit_predict(X_var_filtered)
    bic_scores.append(gmm.bic(X_var_filtered))
    aic_scores.append(gmm.aic(X_var_filtered))
    silhouette_scores.append(silhouette_score(X_var_filtered, labels))

# =============================
# 7. Plot metrici GMM
# =============================
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(n_range, bic_scores, marker='o')
plt.title("BIC")
plt.xlabel("Număr clustere")
plt.grid()

plt.subplot(1, 3, 2)
plt.plot(n_range, aic_scores, marker='x', color='orange')
plt.title("AIC")
plt.xlabel("Număr clustere")
plt.grid()

plt.subplot(1, 3, 3)
plt.plot(n_range, silhouette_scores, marker='s', color='green')
plt.title("Silhouette Score")
plt.xlabel("Număr clustere")
plt.grid()

plt.tight_layout()
plt.show()

# =============================
# 8. Fit final GMM cu k optim
# =============================
optimal_k = np.argmax(silhouette_scores) + 2
print(f"\n✅ Număr optim clustere (după Silhouette): {optimal_k}")

gmm = GaussianMixture(n_components=optimal_k, random_state=42)
df_sample["GMM_Cluster"] = gmm.fit_predict(X_var_filtered)

# =============================
# 9. Analiză pe variabile relevante
# =============================
focus_vars = [
    "History of Mental Illness", "History of Substance Abuse",
    "Family History of Depression", "Chronic Medical Conditions",
    "Smoking Status", "Alcohol Consumption", "Sleep Patterns",
    "Physical Activity Level", "Income"
]
cluster_summary = df_sample.groupby("GMM_Cluster")[focus_vars].mean()
print("\n📊 Medii per cluster pe variabile de sănătate:\n", cluster_summary)

# =============================
# 10. Heatmap pe clustere
# =============================
plt.figure(figsize=(10, 6))
sns.heatmap(cluster_summary, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Medii pe variabile de depresie per cluster")
plt.ylabel("Cluster")
plt.tight_layout()
plt.show()

# =============================
# 11. Vizualizare PCA (2D)
# =============================
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_var_filtered)
df_sample["PCA1"] = X_2d[:, 0]
df_sample["PCA2"] = X_2d[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_sample, x="PCA1", y="PCA2", hue="GMM_Cluster", palette="Set2", alpha=0.6)
plt.title("Vizualizare clustere GMM în spațiu PCA")
plt.grid()
plt.tight_layout()
plt.show()

# =============================
# 12. Identificare cluster depresiv
# =============================
depressive_cluster_id = cluster_summary["History of Mental Illness"].idxmax()
print(f"\n🧠 Clusterul depresiv este: {depressive_cluster_id}")

# =============================
# 13. Profil persoane depresive
# =============================
profile = df_sample[df_sample["GMM_Cluster"] == depressive_cluster_id][selected_features].mean()
print("\n📌 Profil mediu al persoanelor depresive:\n", profile)

# =============================
# 14. Evaluare model final
# =============================
print(f"\n📈 Evaluare model GMM (k = {optimal_k})")
log_likelihood = gmm.score(X_var_filtered) * X_var_filtered.shape[0]
print(f"🔹 Log-Likelihood: {log_likelihood:.2f}")
print(f"🔹 BIC: {gmm.bic(X_var_filtered):.2f}")
print(f"🔹 AIC: {gmm.aic(X_var_filtered):.2f}")
print(f"🔹 Silhouette Score: {silhouette_score(X_var_filtered, df_sample['GMM_Cluster']):.3f}")

# =============================
# 15. Distribuție clustere
# =============================
cluster_counts = df_sample["GMM_Cluster"].value_counts().sort_index()
print("\n🔹 Număr persoane per cluster:\n", cluster_counts)

plt.figure(figsize=(6, 4))
sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="pastel")
plt.title("Distribuția persoanelor pe clustere")
plt.xlabel("Cluster")
plt.ylabel("Număr persoane")
plt.tight_layout()
plt.show()
