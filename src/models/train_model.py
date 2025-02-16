import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Criar diretórios necessários
os.makedirs("src/models/trained_models/", exist_ok=True)

# Carregar os dados processados
df_path = "data/processed/processed_data.csv"
if not os.path.exists(df_path):
    raise FileNotFoundError(f"Arquivo não encontrado: {df_path}")

df = pd.read_csv(df_path)

# Separar features (X) e variável alvo (y)
X = df.drop(columns=["Stress_Level_Biosensor", "Stress_Level_Category"])
y = df["Stress_Level_Category"]

# Normalizar os dados numéricos e salvar o scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "src/models/trained_models/scaler.pkl")

# Dividir os dados em treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# Definir os modelos e seus hiperparâmetros
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "NeuralNetwork": MLPClassifier(max_iter=1000, random_state=42)
}

# Hiperparâmetros para otimização
param_grid = {
    "LogisticRegression": {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["lbfgs", "saga"]
    },
    "RandomForest": {
        "n_estimators": [100, 300],
        "max_depth": [10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 4]
    },
    "NeuralNetwork": {
        "hidden_layer_sizes": [(128, 64), (64, 32)],
        "learning_rate_init": [0.001, 0.01]
    }
}

# Treinar e otimizar os modelos
for name, model in models.items():
    print(f"\n🔍 Otimizando hiperparâmetros para {name}...")
    
    grid_search = GridSearchCV(model, param_grid[name], cv=3, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"✅ Melhor configuração para {name}: {grid_search.best_params_}")

    # Treinar o modelo otimizado
    best_model.fit(X_train, y_train)

    # Salvar o modelo treinado
    joblib.dump(best_model, f"src/models/trained_models/{name}.pkl")

print("\n✅ Modelos treinados e salvos em 'src/models/trained_models/'.")
