import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import seaborn as sns


def kmeansclustering(cleaned_data_path,interpreted_data_path):
    # 1. Încarcă datele direct
    df = pd.read_csv(cleaned_data_path)

    # 2. Eșantion pentru Silhouette Score 
    sample_df = df.sample(n=20000, random_state=42)

    # 3. Găsim k optim
    silhouette_scores = []
    cluster_range = range(2, 11)

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(sample_df)
        silhouette_scores.append(silhouette_score(sample_df, kmeans.labels_))

    # 4. Plot Silhouette Score
    plt.figure(figsize=(8, 5))
    plt.plot(cluster_range, silhouette_scores, marker='o', color='green')
    plt.title("Scorul Silhouette (eșantion de 10.000)")
    plt.xlabel("Număr de clustere")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.show()

    # 5. Alege k în funcție de graficul anterior (ex: k = 4)
    k_optimal = 4 

    # 6. Aplicăm KMeans pe tot datasetul
    kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(df)

    # 7. Analiză pe clustere
    summary = df.groupby("Cluster").mean()
    print("Medii pe fiecare cluster:\n", summary)
    summary.to_csv(interpreted_data_path, index=True)

    # 8. Vizualizare cu PCA
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df.drop(columns=["Cluster"]))
    df["PCA1"] = df_pca[:, 0]
    df["PCA2"] = df_pca[:, 1]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette="Set2")
    plt.title("Clustere vizualizate în 2D (PCA)")
    plt.show()