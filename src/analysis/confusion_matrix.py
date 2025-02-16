import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Caminhos dos modelos treinados
models_paths = {
    "LogisticRegression": "src/models/trained_models/LogisticRegression.pkl",
    "RandomForest": "src/models/trained_models/RandomForest.pkl",
    "NeuralNetwork": "src/models/trained_models/NeuralNetwork.pkl"
}

# Carregar o scaler treinado para garantir que os dados tenham a mesma escala
scaler_path = "src/models/trained_models/scaler.pkl"
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"❌ O scaler não foi encontrado! Execute 'train_model.py' primeiro.")

scaler = joblib.load(scaler_path)

# Verificar se todos os modelos existem
for model_name, model_path in models_paths.items():
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ O modelo {model_name} não foi encontrado! Certifique-se de que 'train_model.py' foi executado corretamente.")

# Carregar os modelos treinados
models = {name: joblib.load(path) for name, path in models_paths.items()}

# Carregar os dados processados
df = pd.read_csv("data/processed/processed_data.csv")

# Separar features (X) e variável alvo (y)
X = df.drop(columns=["Stress_Level_Biosensor", "Stress_Level_Category"])
y = df["Stress_Level_Category"]

# Aplicar a mesma normalização dos dados
X_scaled = scaler.transform(X)

confusion_matrices = {}

for model_name, model in models.items():
    y_pred = model.predict(X_scaled)
    cm = confusion_matrix(y, y_pred, labels=[0, 1, 2])
    confusion_matrices[model_name] = cm

    report = classification_report(y, y_pred, target_names=["Baixo", "Médio", "Alto"], output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    report_path = f"src/reports/model_performance/classification_report_{model_name}.csv"
    report_df.to_csv(report_path)
    print(f"✅ Relatório de classificação para {model_name} salvo em '{report_path}'")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (model_name, cm) in enumerate(confusion_matrices.items()):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Baixo", "Médio", "Alto"], 
                yticklabels=["Baixo", "Médio", "Alto"], ax=axes[i])
    axes[i].set_title(f"Matriz de Confusão - {model_name}")
    axes[i].set_xlabel("Previsto")
    axes[i].set_ylabel("Real")

plt.savefig("src/reports/figures/confusion_matrix_all_models.png", bbox_inches="tight")

print("\n✅ Matrizes de confusão geradas e salvas.")
