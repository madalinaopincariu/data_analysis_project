import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def create_visualizations(data):
    # Dacă data nu este un DataFrame, încercăm să îl convertim
    if not isinstance(data, pd.DataFrame):
        try:
            # Dacă este un fișier CSV sau un alt tip de date, îl convertim într-un DataFrame
            data = pd.read_csv(data)
        except Exception as e:
            raise TypeError(f"Nu am putut converte datele într-un DataFrame: {e}")

    # Creăm un dictionar pentru valorile originale (pentru vizualizare)
    original_values = {
        'Marital Status': {'single': 0, 'married': 1, 'divorced': 2, 'widowed': 3},
        'Education Level': {"associate degree": 0, "bachelor's degree": 1, "high school": 2, "master's degree": 3, "phd": 4},
        'Smoking Status': {"current": 0, "former": 1, "non-smoker": 2},
        'Physical Activity Level': {"active": 0, "moderate": 1, "sedentary": 2},
        'Employment Status': {"employed": 0, "unemployed": 1},
        'Alcohol Consumption': {"high": 0, "low": 1, "moderate": 2},
        'Dietary Habits': {"moderate": 0, "healthy": 1, "unhealthy": 2},
        'Sleep Patterns': {"fair": 0, "good": 1, "poor": 2},
        'History of Mental Illness': {"yes": 0, "no": 1},
        'History of Substance Abuse': {"yes": 0, "no": 1},
        'Family History of Depression': {"yes": 0, "no": 1},
        'Chronic Medical Conditions': {"yes": 0, "no": 1}
    }

    # 1. Normalizarea datelor categorice (Label Encoding)
    label_encoder = LabelEncoder()
    data['Employment Status'] = label_encoder.fit_transform(data['Employment Status'])
    data['History of Substance Abuse'] = label_encoder.fit_transform(data['History of Substance Abuse'])
    data['History of Mental Illness'] = label_encoder.fit_transform(data['History of Mental Illness'])
    data['Marital Status'] = label_encoder.fit_transform(data['Marital Status'])

    # 2. Histograma vârstei
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Age'], kde=True, color="skyblue", bins=15)
    plt.title('Distribuția vârstei')
    plt.xlabel('Vârstă')
    plt.ylabel('Frecvență')
    plt.show()

    # 3. Boxplot pentru venit
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=data['Income'], color="lightgreen")
    plt.title('Distribuția venitului')
    plt.xlabel('Venit')
    plt.show()

    # 4. Scatter plot pentru Vârsta vs Venitul anual
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data['Age'], y=data['Income'], color="purple")
    plt.title('Vârsta vs Venitul')
    plt.xlabel('Vârstă')
    plt.ylabel('Venit')
    plt.show()

    # 5. Heatmap pentru corelațiile între variabilele numerice
    numeric_data = data.select_dtypes(include=['number'])  # Selectează doar coloanele numerice
    correlation_matrix = numeric_data.corr()  # Calculează matricea de corelație
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
    plt.title("Matricea de Corelație")
    plt.show()

    # 6. Vizualizarea distribuției Employment Status folosind datele normalizate, dar etichete nenormalizate pe axă
    plt.figure(figsize=(8, 6))
    sns.countplot(x=data['Employment Status'], palette="muted")
    plt.title("Distribuția Statutului de Angajare (Employment Status)")
    plt.xlabel("Statutul de Angajare (Employment Status)")
    plt.ylabel("Număr de Persoane")
    plt.xticks(ticks=[0, 1], labels=list(original_values['Employment Status'].values()))
    plt.show()

    # 7. Factorii de risc (Abuzul de substanțe vs Boală mentală) folosind datele normalizate
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x='History of Substance Abuse', hue='History of Mental Illness', palette='Set1')
    plt.title('Abuzul de substanțe vs Boală mentală')
    plt.xlabel('Abuz de substanțe')
    plt.ylabel('Număr persoane')
    plt.xticks(ticks=[0, 1], labels=list(original_values['History of Substance Abuse'].values()))
    plt.legend(title='Boală mentală', labels=list(original_values['History of Mental Illness'].values()))
    plt.show()

    # 8. Numărul de copii vs Statusul marital folosind datele normalizate, dar etichete nenormalizate pe axă
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='Number of Children', y='Marital Status', hue='Marital Status', palette='Set2')
    plt.title('Numărul de copii vs Statusul marital')
    plt.xlabel('Număr copii')
    plt.ylabel('Status marital')
    plt.legend(title='Status Marital', labels=list(original_values['Marital Status'].values()))
    plt.show()