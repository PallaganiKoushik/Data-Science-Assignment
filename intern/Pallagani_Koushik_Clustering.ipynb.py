import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
customers_df = pd.read_csv('Customers.csv')
transactions_df = pd.read_csv('Transactions.csv')

# Merge datasets
merged_df = transactions_df.merge(customers_df, on='CustomerID')

# Feature engineering
customer_features = merged_df.groupby('CustomerID').agg({
    'TotalValue': 'sum',   # Total transaction value
    'Quantity': 'sum',     # Total quantity purchased
    'Region': 'first',     # Region of the customer
    'SignupDate': 'first', # Signup date
}).reset_index()

# Encode categorical variables (Region)
customer_features = pd.get_dummies(customer_features, columns=['Region'], drop_first=True)

# Convert SignupDate to numerical days since first date
customer_features['SignupDate'] = pd.to_datetime(customer_features['SignupDate'])
customer_features['DaysSinceSignup'] = (pd.Timestamp.now() - customer_features['SignupDate']).dt.days
customer_features.drop(columns=['SignupDate'], inplace=True)

# Normalize numerical features
scaler = StandardScaler()
numeric_cols = ['TotalValue', 'Quantity', 'DaysSinceSignup']
customer_features[numeric_cols] = scaler.fit_transform(customer_features[numeric_cols])

# Choose the number of clusters (2-10)
db_index_list = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(customer_features.drop(columns=['CustomerID']))
    db_index = davies_bouldin_score(customer_features.drop(columns=['CustomerID']), clusters)
    silhouette = silhouette_score(customer_features.drop(columns=['CustomerID']), clusters)
    db_index_list.append(db_index)
    silhouette_scores.append(silhouette)

# Plot DB Index and Silhouette Scores
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(k_range, db_index_list, marker='o', label='DB Index')
plt.title('DB Index for Different Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('DB Index')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, marker='o', label='Silhouette Score')
plt.title('Silhouette Scores for Different Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.legend()
plt.tight_layout()
plt.show()

# Optimal clusters (based on minimum DB Index)
optimal_k = k_range[np.argmin(db_index_list)]
print(f"Optimal number of clusters based on DB Index: {optimal_k}")

# Perform K-Means clustering with optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
customer_features['Cluster'] = kmeans.fit_predict(customer_features.drop(columns=['CustomerID']))

# Final clustering metrics
db_index = davies_bouldin_score(customer_features.drop(columns=['CustomerID', 'Cluster']), customer_features['Cluster'])
silhouette = silhouette_score(customer_features.drop(columns=['CustomerID', 'Cluster']), customer_features['Cluster'])

print(f"Davies-Bouldin Index: {db_index}")
print(f"Silhouette Score: {silhouette}")

# Visualization of clusters (using PCA for dimensionality reduction)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_features = pca.fit_transform(customer_features.drop(columns=['CustomerID', 'Cluster']))
pca_df = pd.DataFrame(pca_features, columns=['PCA1', 'PCA2'])
pca_df['Cluster'] = customer_features['Cluster']

plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue='Cluster', palette='tab10')
plt.title('Clusters Visualized (PCA Reduced to 2D)')
plt.show()

# Save clustering results
customer_features.to_csv('Customer_Clusters.csv', index=False)
print("Clustering completed! Results saved to 'Customer_Clusters.csv'.")
