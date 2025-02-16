import pandas as pd

# Carregar o dataset processado
df = pd.read_csv("data/processed/processed_data.csv")

# Exibir os nomes das colunas disponíveis
print("\n📌 Colunas no dataset processado:\n")
print(df.columns)
