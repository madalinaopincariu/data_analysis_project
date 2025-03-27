import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_and_normalize_data(input_path, output_path):
    # Încărcare dataset
    df = pd.read_csv(input_path)
    
    # Eliminarea valorilor lipsă
    df.dropna(inplace=True)
    
    # Filtrare: păstrează doar rândurile cu Age ≤ 65
    df = df[df['Age'] <= 65]
    
    # Conversia coloanelor numerice la tipul corect
    df['Age'] = df['Age'].astype(int)
    df['Income'] = df['Income'].astype(float)
    df['Number of Children'] = df['Number of Children'].astype(int)
    
    # Definire mapping pentru encoding manual
    encoding_maps = {
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
    
    # Aplicare encoding manual
    for col, mapping in encoding_maps.items():
        df[col] = df[col].str.lower().map(mapping)
    
    # Normalizarea coloanelor numerice
    scaler = StandardScaler()
    df[['Age', 'Income', 'Number of Children']] = scaler.fit_transform(df[['Age', 'Income', 'Number of Children']])
    
    # Salvarea datelor curățate
    df.to_csv(output_path, index=False)
    print(f"Data cleaned and saved to {output_path}")
