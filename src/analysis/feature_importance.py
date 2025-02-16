import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Caminho do modelo treinado
model_path = "src/models/trained_models/RandomForest.pkl"

# Verificar se o modelo existe
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Modelo não encontrado: {model_path}. Verifique o caminho ou reexecute o treinamento.")

# Carregar o modelo treinado
rf_model = joblib.load(model_path)

# Carregar os dados processados
df = pd.read_csv("data/processed/processed_data.csv")

# Criar a coluna `Stress_Level_Category` se ela não existir
if "Stress_Level_Category" not in df.columns:
    print("⚠️ Coluna 'Stress_Level_Category' não encontrada. Criando agora...")
    df["Stress_Level_Category"] = pd.qcut(df["Stress_Level_Biosensor"], q=3, labels=[0, 1, 2])

# Remover as colunas relacionadas ao estresse para evitar vazamento de dados
X = df.drop(columns=["Stress_Level_Biosensor", "Stress_Level_Self_Report"])

# Obter importância das features
feature_importance = rf_model.feature_importances_
features = X.columns

# Criar DataFrame com as importâncias
importance_df = pd.DataFrame({"Feature": features, "Importance": feature_importance})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Exibir DataFrame
print("\n📊 Importância das Features no Modelo RandomForest:\n")
print(importance_df)

# Criar gráfico de barras para importância das features
plt.figure(figsize=(12, 6))
plt.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
plt.xlabel("Importância")
plt.ylabel("Feature")
plt.title("Importância das Features no RandomForest")
plt.gca().invert_yaxis()  # Inverter eixo para a mais importante ficar no topo
plt.savefig("src/reports/figures/feature_importance.png", bbox_inches="tight")  # Salvar gráfico
#plt.show()
