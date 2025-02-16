# 📘 Predição de Estresse Acadêmico com Aprendizado de Máquina

Este projeto tem como objetivo desenvolver modelos de aprendizado de máquina para prever os níveis de estresse em estudantes universitários, analisando fatores como qualidade do sono, atividade física e parâmetros fisiológicos.

---

## 📌 Sobre o Projeto

O estresse acadêmico é um problema crescente em instituições de ensino superior, impactando o desempenho e a saúde mental dos estudantes. Utilizando técnicas de aprendizado de máquina, este projeto busca identificar padrões e prever níveis de estresse, possibilitando a implementação de estratégias para mitigação e promoção do bem-estar dos alunos.

---

## 📊 Metodologia

O estudo foi conduzido utilizando uma base de dados composta por informações de estudantes universitários, considerando:

- **Variáveis analisadas:** Qualidade do sono, atividade física, estado emocional, carga horária de estudos, frequência cardíaca e pressão arterial.
- **Modelos de aprendizado de máquina testados:** 
  - Regressão Logística
  - Random Forest
  - Rede Neural Artificial
- **Métricas de avaliação:** Acurácia, Precisão, Recall e F1-Score.

Os dados foram tratados, normalizados e analisados estatisticamente para garantir resultados confiáveis.

---

## 🚀 Tecnologias Utilizadas

- **Linguagem:** Python 🐍
- **Bibliotecas principais:**
  - `pandas` → Manipulação e limpeza de dados
  - `numpy` → Operações matemáticas
  - `matplotlib` e `seaborn` → Visualização de dados
  - `scikit-learn` → Modelagem e avaliação de aprendizado de máquina
  - `TensorFlow/Keras` → Treinamento de redes neurais

---

## 📂 Estrutura do Projeto

```bash
📁 stress-prediction
│── 📂 data                 # Conjunto de dados utilizados
│── 📂 notebooks            # Notebooks Jupyter com as análises e treinamentos
│── 📂 models               # Modelos treinados e serializados
│── 📂 illustrations        # Imagens e gráficos gerados
│── 📄 utfpr-article.tex    # Artigo em LaTeX descrevendo o estudo
│── 📄 requirements.txt     # Dependências do projeto
│── 📄 README.md            # Documentação do projeto
```

---

## 🔬 Resultados Obtidos

Os testes indicaram que o modelo **Random Forest** apresentou a melhor performance, com acurácia superior a **90%**, seguido pela **Rede Neural**. Já a **Regressão Logística** teve um desempenho inferior, com métricas em torno de **55%**.

As principais descobertas incluem:

✔️ **Correlação entre qualidade do sono e estresse:** Estudantes com padrões de sono inadequados apresentaram níveis elevados de estresse.  
✔️ **Impacto da atividade física:** Indivíduos que praticam exercícios regularmente tendem a apresentar menores níveis de estresse.  
✔️ **Modelos de aprendizado de máquina como ferramenta de monitoramento:** As previsões podem ser usadas para antecipar episódios de estresse e recomendar estratégias de mitigação.  

---

## ⚙️ Como Executar o Projeto

1. Clone este repositório:
   ```bash
   git clone https://github.com/oBaldon/stress-prediction.git
   ```
2. Acesse a pasta do projeto:
   ```bash
   cd stress-prediction
   ```
3. Crie um ambiente virtual (opcional, mas recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```
4. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
5. Execute os notebooks para visualizar as análises e os modelos:
   ```bash
   jupyter notebook
   ```

---

## 🛠️ Próximos Passos

✔️ Melhorar a coleta de dados com mais variáveis (ex.: alimentação, uso de dispositivos eletrônicos antes do sono).  
✔️ Implementar técnicas de **aprendizado profundo** para aprimorar a precisão dos modelos.  
✔️ Desenvolver um **dashboard interativo** para visualização dos resultados em tempo real.  

---

## 📜 Referências

As principais referências utilizadas na fundamentação teórica do projeto estão disponíveis no artigo acadêmico desenvolvido. Caso tenha interesse em consultar os estudos citados, acesse o arquivo `utfpr-article.bib`.

---

## 🤝 Contribuição

Ficaremos felizes com qualquer contribuição para aprimorar este estudo! Caso tenha sugestões ou melhorias:

1. Faça um **fork** do repositório.
2. Crie uma nova branch:
   ```bash
   git checkout -b feature-nova-melhoria
   ```
3. Faça suas modificações e **commit**:
   ```bash
   git commit -m "Melhoria na análise dos dados"
   ```
4. Envie para o repositório remoto:
   ```bash
   git push origin feature-nova-melhoria
   ```
5. Abra um **Pull Request** 🚀

---

## 📧 Contato

📌 Desenvolvedor: **Douglas Baldon Corrêa**  
📧 E-mail: douglasbaldon@alunos.utfpr.edu.br  
🔗 GitHub: [oBaldon](https://github.com/oBaldon)

Se tiver dúvidas ou sugestões, entre em contato! 😊

---

> Projeto desenvolvido como parte de um estudo acadêmico na Universidade Tecnológica Federal do Paraná (UTFPR).
```

---
