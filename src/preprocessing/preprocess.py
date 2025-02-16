import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

def load_data(file_path):
    """Carrega o dataset a partir de um arquivo CSV.""" 
    return pd.read_csv(file_path)

def clean_data(df):
    """Realiza limpeza de dados, removendo duplicatas e preenchendo valores ausentes apenas nas colunas numéricas.""" 
    df = df.drop_duplicates()

    # Preencher apenas as colunas numéricas com a média
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    return df

def encode_categorical(df, categorical_columns):
    """Transforma variáveis categóricas usando One-Hot Encoding.""" 
    encoder = OneHotEncoder(sparse_output=False, drop="first")
    encoded_features = encoder.fit_transform(df[categorical_columns])
    feature_names = encoder.get_feature_names_out(categorical_columns)
    encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=df.index)
    df = df.drop(columns=categorical_columns).reset_index(drop=True)
    return pd.concat([df, encoded_df], axis=1)

def create_stress_category(df):
    """Cria a coluna `Stress_Level_Category` com base na variável `Stress_Level_Biosensor`."""
    if "Stress_Level_Biosensor" in df.columns:
        df["Stress_Level_Category"] = pd.qcut(df["Stress_Level_Biosensor"], q=3, labels=[0, 1, 2])
        df["Stress_Level_Category"] = df["Stress_Level_Category"].astype(int)
    else:
        raise KeyError("A coluna 'Stress_Level_Biosensor' não está presente no dataset.")
    return df

def normalize_numerical(df, numerical_columns):
    """Normaliza variáveis numéricas usando Min-Max Scaling.""" 
    scaler = MinMaxScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df

def preprocess_pipeline():
    """Pipeline completo de pré-processamento.""" 
    file_path = "data/raw/student_health_data.csv"
    df = load_data(file_path)
    df = clean_data(df)

    categorical_columns = ["Gender", "Physical_Activity", "Sleep_Quality", "Mood", "Health_Risk_Level"]
    numerical_columns = ["Age", "Heart_Rate", "Blood_Pressure_Systolic", "Blood_Pressure_Diastolic", 
                         "Stress_Level_Biosensor", "Stress_Level_Self_Report", "Study_Hours", "Project_Hours"]

    df = encode_categorical(df, categorical_columns)
    df = create_stress_category(df)  # ✅ Garante que a coluna seja criada antes de normalizar
    df = normalize_numerical(df, numerical_columns)

    output_path = "data/processed/processed_data.csv"
    df.to_csv(output_path, index=False)

    print(f"✅ Dataset processado salvo em: {output_path}")
    print(f"📌 Colunas no dataset final: {df.columns}")

if __name__ == "__main__":
    preprocess_pipeline()
