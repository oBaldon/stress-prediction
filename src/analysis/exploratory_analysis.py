import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Criar diretório para salvar os gráficos
output_dir = "src/reports/figures"
os.makedirs(output_dir, exist_ok=True)

# Carregar o dataset processado
df = pd.read_csv("data/processed/processed_data.csv")

# Configurar estilo dos gráficos
sns.set_theme(style="whitegrid")

# Histograma - Distribuição dos Níveis de Estresse
plt.figure(figsize=(10, 5))
sns.histplot(df["Stress_Level_Biosensor"], bins=20, kde=True, color="blue", label="Biossensor")
sns.histplot(df["Stress_Level_Self_Report"], bins=20, kde=True, color="red", label="Auto-Relato")
plt.xlabel("Nível de Estresse")
plt.ylabel("Frequência")
plt.title("Distribuição dos Níveis de Estresse")
plt.legend()
plt.savefig(f"{output_dir}/stress_distribution.png", bbox_inches="tight")
plt.close()

# Mapa de Correlação
plt.figure(figsize=(14, 10))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.title("Mapa de Correlação entre as Variáveis")
plt.tight_layout()
plt.savefig(f"{output_dir}/correlation_matrix.png", bbox_inches="tight")
plt.close()

# Boxplot - Relação entre Qualidade do Sono e Estresse
plt.figure(figsize=(10, 5))
sns.boxplot(x=df["Sleep_Quality_Moderate"], y=df["Stress_Level_Biosensor"], hue=df["Sleep_Quality_Moderate"], palette="coolwarm", legend=False)
plt.xlabel("Qualidade do Sono (0 = Ruim, 1 = Moderada)")
plt.ylabel("Nível de Estresse (Biossensor)")
plt.title("Relação entre Qualidade do Sono e Estresse")
plt.savefig(f"{output_dir}/sleep_vs_stress.png", bbox_inches="tight")
plt.close()

# Boxplot - Relação entre Atividade Física e Estresse
plt.figure(figsize=(10, 5))
sns.boxplot(x=df["Physical_Activity_Moderate"], y=df["Stress_Level_Biosensor"], hue=df["Physical_Activity_Moderate"], palette="coolwarm", legend=False)
plt.xlabel("Atividade Física (0 = Baixa, 1 = Moderada)")
plt.ylabel("Nível de Estresse (Biossensor)")
plt.title("Relação entre Atividade Física e Estresse")
plt.savefig(f"{output_dir}/activity_vs_stress.png", bbox_inches="tight")
plt.close()

print(f"Gráficos salvos em: {output_dir}")
