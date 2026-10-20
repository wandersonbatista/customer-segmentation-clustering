# Segmentação de Clientes Varejistas (Clustering)

Este projeto de Ciência de Dados resolve um problema clássico de marketing estratégico: como identificar grupos distintos de consumidores dentro de uma base genérica de clientes. Utilizando algoritmos de Machine Learning Não-Supervisionado, os clientes foram divididos em perfis acionáveis, baseados no comportamento real em vez de intuição.

## O Desafio
Campanhas de marketing genéricas (disparar o mesmo e-mail para todos) são ineficientes e caras. O objetivo deste projeto é fornecer à equipe de negócios uma separação clara dos clientes para que campanhas possam ser altamente personalizadas. A pergunta respondida foi: "Quantos perfis de comportamento de compra existem em nossa base e quem são eles?".

## Tecnologias Utilizadas
* **Linguagem:** Python
* **Tratamento de Dados:** Pandas, NumPy
* **Visualização:** Seaborn, Matplotlib
* **Machine Learning:** Scikit-Learn
* **Algoritmo:** K-Means
* **Métricas de Validação:** Método do Cotovelo (Elbow Method) e Silhouette Score

## Arquitetura da Solução

1. **Pré-processamento (Scaling):** Como algoritmos de clusterização utilizam a Distância Euclidiana, as variáveis `Annual Income` (Renda Anual) e `Spending Score` (Pontuação de Gasto) foram normalizadas utilizando o `StandardScaler` para garantir que ambas tivessem o mesmo peso na decisão do algoritmo.
2. **Definição Matemática dos Clusters:** Para fugir do achismo na escolha do número de grupos, foram executados loops de testes. O gráfico do **Método do Cotovelo** cruzado com a avaliação do **Silhouette Score** apontou matematicamente que `k=5` era a divisão perfeita para os dados.
3. **Treinamento e Visualização:** O modelo dividiu a base com clareza cristalina. Uma visualização de dispersão foi gerada contendo a localização exata dos centroides geométricos.

## Conclusões Extraídas (Insights)
O modelo isolou 5 arquétipos de consumidores:
*   **Alta Renda, Alto Gasto:** Os clientes ideais. Público para lançamentos premium.
*   **Alta Renda, Baixo Gasto:** Oportunidade perdida. Têm dinheiro, mas compram pouco. Exigem pesquisa de mercado.
*   **Baixa Renda, Alto Gasto:** Compradores impulsivos. Alvo ideal para liquidações e cupons de desconto.
*   **Baixa Renda, Baixo Gasto:** Clientes esporádicos.
*   **Renda Média, Gasto Médio:** A grande massa de clientes regulares.

## 🚀 Como Executar

1. Clone o repositório,

2. Instale as dependências:

    `pip install pandas numpy matplotlib seaborn scikit-learn`

3. Baixe o Mall Customers Dataset e adicione à raiz do projeto,

4. Execute o arquivo `csc.py`.