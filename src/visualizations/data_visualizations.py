import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def create_visualizations(cleaned_data_path, raw_data_path):
    # Încărcăm datele
    data_cleaned = pd.read_csv(cleaned_data_path)
    data_raw = pd.read_csv(raw_data_path)

    # Eliminăm valorile lipsă și filtrăm vârsta ≤ 65
    data_raw.dropna(inplace=True)
    data_raw = data_raw[data_raw['Age'] <= 65]

    # Verificare duplicat
    if data_raw.duplicated().sum() > 0:
        data_raw.drop_duplicates(inplace=True)
    
    # Verificare outliers: eliminăm valorile extreme folosind IQR
    Q1 = data_raw['Income'].quantile(0.25)
    Q3 = data_raw['Income'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data_raw = data_raw[(data_raw['Income'] >= lower_bound) & (data_raw['Income'] <= upper_bound)]

    # Verificăm dacă fișierele au același număr de rânduri
    if len(data_cleaned) != len(data_raw):
        raise ValueError("Fișierele cleaned.csv și raw.csv nu au același număr de rânduri!")

    #Histogramă pentru distribuția vârstei
    plt.figure(figsize=(10, 6))
    sns.histplot(data_raw['Age'], kde=True, color="skyblue", bins=15)
    plt.title('Distribuția vârstei')
    plt.xlabel('Vârstă')
    plt.ylabel('Frecvență')
    plt.show()

    #Boxplot pentru distribuția venitului
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=data_raw['Income'], color="lightgreen")
    plt.title('Distribuția venitului')
    plt.xlabel('Venit')
    plt.show()

    #Countplot pentru Employment Status
    plt.figure(figsize=(8, 6))
    sns.countplot(x=data_raw['Employment Status'], palette="muted")
    plt.title("Distribuția Statutului de Angajare")
    plt.xlabel("Statutul de Angajare")
    plt.ylabel("Număr de Persoane")
    plt.xticks(rotation=45)
    plt.show()

    #Countplot pentru Abuzul de Substanțe vs. Boală Mentală
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data_raw, x='History of Substance Abuse', hue='History of Mental Illness', palette='Set1')
    plt.title('Abuzul de substanțe vs Boală mentală')
    plt.xlabel('Abuz de substanțe')
    plt.ylabel('Număr persoane')
    plt.legend(title='Boală mentală')
    plt.show()