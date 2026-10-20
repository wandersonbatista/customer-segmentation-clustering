import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

sns.set_theme(style="whitegrid")

# Carregando o dataset (certifique-se de que o CSV está na mesma pasta)
df = pd.read_csv('Mall_Customers.csv')
print(df.head())


# Removendo a coluna de ID, que não ajuda na clusterização
df_cluster = df.drop('CustomerID', axis=1)

# Visualizando a dispersão inicial
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_cluster, x='Annual Income (k$)', y='Spending Score (1-100)', color='grey', alpha=0.6)
plt.title('Renda Anual vs Pontuação de Gastos (Sem Clusters)')
plt.show()


# Selecionando apenas as variáveis numéricas de interesse para este teste (Renda e Gasto)
X = df_cluster[['Annual Income (k$)', 'Spending Score (1-100)']]

# Escalonamento
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Dados escalonados com sucesso.")


wcss = []
silhouette_scores = []

for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plotando o Método do Cotovelo
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(2, 11), wcss, marker='o', linestyle='--')
plt.title('Método do Cotovelo (Elbow Method)')
plt.xlabel('Número de Clusters')
plt.ylabel('WCSS (Inércia)')

# Plotando o Silhouette Score (O quão bem separados os clusters estão. Quanto mais perto de 1, melhor)
plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='--', color='green')
plt.title('Silhouette Score')
plt.xlabel('Número de Clusters')
plt.ylabel('Score')

plt.tight_layout()
plt.show()


# Treinando o modelo definitivo
kmeans_final = KMeans(n_clusters=5, init='k-means++', random_state=42, n_init=10)
y_kmeans = kmeans_final.fit_predict(X_scaled)

# Adicionando os rótulos dos clusters de volta ao DataFrame original
df['Cluster'] = y_kmeans
print(df.head())


plt.figure(figsize=(10, 6))

# Plotando os dados coloridos pelo cluster atribuído
sns.scatterplot(
    x='Annual Income (k$)', 
    y='Spending Score (1-100)', 
    hue='Cluster', 
    data=df, 
    palette='Set1', 
    s=100, 
    alpha=0.8
)

# Plotando os centroides (o "coração" matemático de cada grupo)
centroides = scaler.inverse_transform(kmeans_final.cluster_centers_)
plt.scatter(centroides[:, 0], centroides[:, 1], s=300, c='black', label='Centroides', marker='X')

plt.title('Segmentação de Clientes (5 Clusters Encontrados)')
plt.xlabel('Renda Anual (k$)')
plt.ylabel('Pontuação de Gastos (1-100)')
plt.legend(title='Grupos (Clusters)')
plt.show()

