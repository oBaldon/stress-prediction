import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Caminhos dos arquivos CSV
csv_paths = {
    "LogisticRegression": "src/reports/model_performance/classification_report_LogisticRegression.csv",
    "RandomForest": "src/reports/model_performance/classification_report_RandomForest.csv",
    "NeuralNetwork": "src/reports/model_performance/classification_report_NeuralNetwork.csv"
}

# Verificar se os arquivos existem
for model, path in csv_paths.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ O arquivo {path} não foi encontrado!")

# Carregar os dados
model_metrics = {}

for model, path in csv_paths.items():
    df = pd.read_csv(path, index_col=0)
    model_metrics[model] = df.loc[["Baixo", "Médio", "Alto"], ["precision", "recall", "f1-score"]]

# Criar gráfico de comparação
metrics = ["precision", "recall", "f1-score"]
categories = ["Baixo", "Médio", "Alto"]
models = list(model_metrics.keys())
x = np.arange(len(categories))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, metric in enumerate(metrics):
    ax = axes[i]
    width = 0.25  # Largura das barras
    for j, model in enumerate(models):
        ax.bar(x + j * width, model_metrics[model][metric], width=width, label=model, alpha=0.8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title(f"Comparação de {metric.capitalize()} entre os Modelos")
    ax.set_ylabel(metric.capitalize())
    ax.legend()

# Criar diretório para salvar os gráficos
os.makedirs("src/reports/figures", exist_ok=True)

# Salvar o gráfico
plt.tight_layout()
plt.savefig("src/reports/figures/model_comparison.png")

# Exibir o gráfico gerado
#plt.show()
