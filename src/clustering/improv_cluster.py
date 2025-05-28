import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def kmeansclustering(
    cleaned_data_path,
    interpreted_data_path,
    max_k=12,
    apply_pca_before_kmeans=False,
    save_plots=False,
    sample_size=20000
):
    # 1. Încărcare
    df = pd.read_csv(cleaned_data_path)
    print(f"✅ Date inițiale: {df.shape}")

    # 2. Sampling pentru selecția lui k
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    print(f"🔍 Eșantion pentru alegerea lui k: {df_sample.shape}")

    # 3. PCA 
    if apply_pca_before_kmeans:
        pca_sample = PCA(n_components=0.95)
        sample_processed = pca_sample.fit_transform(df_sample)
        full_processed = pca_sample.transform(df)  # aplicăm același PCA și pe tot setul
        print(f"🔧 PCA aplicat: {sample_processed.shape}")
    else:
        sample_processed = df_sample.values
        full_processed = df.values

    # 4. Găsire k optim (pe sample)
    silhouette_scores = []
    wcss_scores = []

    cluster_range = range(2, 10)
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(sample_processed)
        silhouette_scores.append(silhouette_score(sample_processed, labels))
        wcss_scores.append(kmeans.inertia_)

    # 5. Plot metrici
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(cluster_range, silhouette_scores, marker='o')
    axs[0].set_title("Silhouette Score (Sample)")
    axs[0].set_xlabel("Număr de clustere")
    axs[0].set_ylabel("Silhouette")
    axs[0].grid()

    axs[1].plot(cluster_range, wcss_scores, marker='x', color='red')
    axs[1].set_title("Elbow Method (Sample)")
    axs[1].set_xlabel("Număr de clustere")
    axs[1].set_ylabel("WCSS")
    axs[1].grid()

    plt.tight_layout()
    plt.show()

    # 6. k optim ales pe baza silhouette score
    k_optimal = cluster_range[np.argmax(silhouette_scores)]
    print(f"\n✅ k optim (pe eșantion): {k_optimal}")

    # 7. Fit final pe întregul set
    final_kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
    df["Cluster"] = final_kmeans.fit_predict(full_processed)

    # 8. Analiză clustere
    summary = df.groupby("Cluster").mean()
    print("\n📊 Medii pe fiecare cluster:\n", summary.round(2))
    summary.to_csv(interpreted_data_path, index=True)

    # 9. Vizualizare cu PCA 2D
    pca_vis = PCA(n_components=2)
    X_vis = pca_vis.fit_transform(full_processed)
    df["PCA1"] = X_vis[:, 0]
    df["PCA2"] = X_vis[:, 1]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette="Set2", alpha=0.7)
    plt.title("Vizualizare PCA (2D) a clusterelor")
    plt.show()

    # 10. Evaluare finală
    sil_score_final = silhouette_score(
    full_processed,
    df["Cluster"],
    sample_size=10000,
    random_state=42
)

    print(f"\n📈 Scor Silhouette final pe întregul set: {sil_score_final:.3f}")

# Exemplu de apel
kmeansclustering(
    cleaned_data_path="data/cleaned/cleaned_data.csv",
    interpreted_data_path="data/improv2/kmeans_summary.csv",
    max_k=10,
    apply_pca_before_kmeans=True,
    save_plots=True,
    sample_size=10000  # Eșantion pentru selecție k
)
