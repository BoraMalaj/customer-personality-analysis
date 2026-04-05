#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('marketing_campaign.csv', sep='\t')

print("Dataset loaded successfully!")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
df.head()


# In[21]:


# Check missing values
print("Missing Values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Drop missing values
df = df.dropna()
print("\nShape after dropping missing values:", df.shape)


# In[23]:


# Create Age feature
df['Age'] = 2025 - df['Year_Birth']

# Select features for clustering
features = ['Age', 'Income', 'MntWines', 'MntFruits', 'MntMeatProducts',
            'MntFishProducts', 'MntSweetProducts', 'NumWebPurchases',
            'NumStorePurchases', 'NumDealsPurchases']

df_kmeans = df[features].copy()
print("Features ready!")
print(df_kmeans.describe().round(2))


# In[25]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_kmeans)

print("Data scaled successfully!")
print("Shape:", df_scaled.shape)


# In[27]:


from sklearn.cluster import KMeans

inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, marker='o', color='steelblue', linewidth=2)
plt.title('Elbow Method - Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.xticks(k_range)
plt.grid(linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# In[29]:


optimal_k = 4

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(df_scaled)

print(f"K-Means trained with K={optimal_k}")
print("\nCluster Distribution:")
print(df['cluster'].value_counts().sort_index())


# In[31]:


plt.figure(figsize=(10, 6))
colors = ['steelblue', 'green', 'orange', 'red']

for cluster in range(optimal_k):
    subset = df[df['cluster'] == cluster]
    plt.scatter(subset['Income'], subset['MntWines'],
                label=f'Cluster {cluster}',
                color=colors[cluster],
                alpha=0.6, edgecolor='black', s=50)

plt.title('K-Means Clusters - Income vs Wine Spending')
plt.xlabel('Income')
plt.ylabel('Wine Spending')
plt.legend()
plt.grid(linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# In[33]:


print("Cluster Profiles - Mean Values:")
print("=" * 60)
print(df.groupby('cluster')[features].mean().round(2).to_string())


# In[35]:


from sklearn.metrics import silhouette_score

score = silhouette_score(df_scaled, df['cluster'])
print(f"Silhouette Score: {round(score, 2)}")
print("\nInterpretation:")
print("  0.7 - 1.0 → Strong clusters")
print("  0.5 - 0.7 → Reasonable clusters")
print("  0.3 - 0.5 → Weak clusters")
print("  < 0.3     → Poor clusters")


# In[37]:


from sklearn.neighbors import NearestNeighbors
import numpy as np

# Find optimal epsilon
neighbors = NearestNeighbors(n_neighbors=5)
neighbors.fit(df_scaled)
distances, indices = neighbors.kneighbors(df_scaled)

# Sort distances
distances = np.sort(distances[:, 4], axis=0)

plt.figure(figsize=(10, 6))
plt.plot(distances, color='steelblue', linewidth=2)
plt.title('KNN Distance Plot - Find Optimal Epsilon')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.grid(linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# In[39]:


from sklearn.cluster import DBSCAN

# Train DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
df['dbscan_cluster'] = dbscan.fit_predict(df_scaled)

# Results
n_clusters = len(set(df['dbscan_cluster'])) - (1 if -1 in df['dbscan_cluster'] else 0)
n_noise = list(df['dbscan_cluster']).count(-1)

print(f"Number of Clusters: {n_clusters}")
print(f"Number of Noise Points: {n_noise}")
print(f"\nCluster Distribution:")
print(df['dbscan_cluster'].value_counts().sort_index())


# In[43]:


plt.figure(figsize=(10, 6))

unique_clusters = sorted(df['dbscan_cluster'].unique())
colors = plt.cm.tab10.colors

for cluster in unique_clusters:
    subset = df[df['dbscan_cluster'] == cluster]
    if cluster == -1:
        plt.scatter(subset['Income'], subset['MntWines'],
                   label='Noise', color='black',
                   alpha=0.3, s=30, marker='x')  # removed edgecolor
    else:
        plt.scatter(subset['Income'], subset['MntWines'],
                   label=f'Cluster {cluster}',
                   color=colors[cluster % len(colors)],
                   alpha=0.6, edgecolor='black', s=50)

plt.title('DBSCAN Clusters - Income vs Wine Spending')
plt.xlabel('Income')
plt.ylabel('Wine Spending')
plt.legend()
plt.grid(linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# In[45]:


from sklearn.metrics import silhouette_score

# Only calculate if more than 1 cluster found
if n_clusters > 1:
    # Exclude noise points for silhouette score
    mask = df['dbscan_cluster'] != -1
    score = silhouette_score(df_scaled[mask], df['dbscan_cluster'][mask])
    print(f"Silhouette Score: {round(score, 2)}")
else:
    print("Only 1 cluster found — try adjusting eps or min_samples!")


# In[47]:


fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# K-Means
colors_km = ['steelblue', 'green', 'orange', 'red']
for cluster in range(optimal_k):
    subset = df[df['cluster'] == cluster]
    axes[0].scatter(subset['Income'], subset['MntWines'],
                   label=f'Cluster {cluster}',
                   color=colors_km[cluster],
                   alpha=0.6, edgecolor='black', s=50)
axes[0].set_title('K-Means Clusters')
axes[0].set_xlabel('Income')
axes[0].set_ylabel('Wine Spending')
axes[0].legend()
axes[0].grid(linestyle='--', alpha=0.5)

# DBSCAN
for cluster in unique_clusters:
    subset = df[df['dbscan_cluster'] == cluster]
    if cluster == -1:
        axes[1].scatter(subset['Income'], subset['MntWines'],
                       label='Noise', color='black',
                       alpha=0.3, s=30, marker='x')
    else:
        axes[1].scatter(subset['Income'], subset['MntWines'],
                       label=f'Cluster {cluster}',
                       color=colors[cluster % len(colors)],
                       alpha=0.6, edgecolor='black', s=50)
axes[1].set_title('DBSCAN Clusters')
axes[1].set_xlabel('Income')
axes[1].set_ylabel('Wine Spending')
axes[1].legend()
axes[1].grid(linestyle='--', alpha=0.5)

plt.suptitle('K-Means vs DBSCAN Comparison', fontsize=14)
plt.tight_layout()
plt.show()


# In[49]:


from sklearn.metrics import silhouette_score

# K-Means Silhouette
kmeans_score = silhouette_score(df_scaled, df['cluster'])

# DBSCAN Silhouette (exclude noise)
mask = df['dbscan_cluster'] != -1
dbscan_score = silhouette_score(df_scaled[mask], df['dbscan_cluster'][mask])

print("Cluster Evaluation - Silhouette Score")
print("=" * 40)
print(f"K-Means Silhouette Score  : {round(kmeans_score, 2)}")
print(f"DBSCAN Silhouette Score   : {round(dbscan_score, 2)}")
print("\nInterpretation:")
print("  0.7 - 1.0 → Strong clusters")
print("  0.5 - 0.7 → Reasonable clusters")
print("  0.3 - 0.5 → Weak clusters")
print("  < 0.3     → Poor clusters")

# Winner
if kmeans_score > dbscan_score:
    print(f"\n🏆 K-Means performs better for this dataset!")
else:
    print(f"\n🏆 DBSCAN performs better for this dataset!")


# In[51]:


from sklearn.cluster import KMeans

sse = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_range, sse, marker='o', color='red', linewidth=2)
plt.fill_between(k_range, sse, alpha=0.1, color='red')
plt.title('SSE - Sum of Squared Errors per K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('SSE (Inertia)')
plt.xticks(k_range)
plt.grid(linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

print("SSE Values per K:")
for k, s in zip(k_range, sse):
    print(f"  K={k} → SSE={round(s, 2)}")


# In[53]:


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Elbow
axes[0].plot(k_range, sse, marker='o', color='steelblue', linewidth=2)
axes[0].set_title('Elbow Method')
axes[0].set_xlabel('K')
axes[0].set_ylabel('Inertia')
axes[0].grid(linestyle='--', alpha=0.5)

# SSE
axes[1].plot(k_range, sse, marker='o', color='red', linewidth=2)
axes[1].fill_between(k_range, sse, alpha=0.1, color='red')
axes[1].set_title('SSE per K')
axes[1].set_xlabel('K')
axes[1].set_ylabel('SSE')
axes[1].grid(linestyle='--', alpha=0.5)

# Silhouette
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df_scaled)
    silhouette_scores.append(silhouette_score(df_scaled, labels))

axes[2].plot(range(2, 11), silhouette_scores, marker='o', 
             color='green', linewidth=2)
axes[2].set_title('Silhouette Score per K')
axes[2].set_xlabel('K')
axes[2].set_ylabel('Silhouette Score')
axes[2].grid(linestyle='--', alpha=0.5)

plt.suptitle('Cluster Evaluation - Elbow / SSE / Silhouette', fontsize=14)
plt.tight_layout()
plt.show()

