import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Load datasets
customers_df = pd.read_csv('Customers.csv')
products_df = pd.read_csv('Products.csv')
transactions_df = pd.read_csv('Transactions.csv')

# Merge datasets
merged_df = transactions_df.merge(customers_df, on='CustomerID').merge(products_df, on='ProductID')

# Debugging: Check columns in the merged dataset
print("Columns in merged_df:", merged_df.columns)

# Feature Engineering: Aggregate features at the CustomerID level
customer_features = merged_df.groupby('CustomerID').agg({
    'TotalValue': 'sum',          # Total value of transactions
    'Quantity': 'sum',            # Total quantity purchased
    'Region': 'first',            # Region of the customer
    'ProductID': lambda x: list(x)  # List of products purchased
}).reset_index()

# One-hot encode the Region column
customer_features = pd.get_dummies(customer_features, columns=['Region'])

# Normalize numerical columns
scaler = StandardScaler()
numeric_cols = ['TotalValue', 'Quantity']
customer_features[numeric_cols] = scaler.fit_transform(customer_features[numeric_cols])

# One-hot encode ProductID for each customer
product_dummies = pd.get_dummies(merged_df[['CustomerID', 'ProductID']], columns=['ProductID'], prefix='Product')
customer_product_matrix = product_dummies.groupby('CustomerID').sum()

# Combine all features and drop any problematic columns
final_features = customer_features.set_index('CustomerID').join(customer_product_matrix, on='CustomerID')

# Ensure all features are numeric for cosine similarity
final_features = final_features.apply(pd.to_numeric, errors='coerce').fillna(0)

# Calculate similarity matrix
similarity_matrix = cosine_similarity(final_features)
similarity_df = pd.DataFrame(similarity_matrix, index=final_features.index, columns=final_features.index)

# Generate Lookalike Results
lookalike_results = {}
for customer_id in final_features.index[:20]:  # First 20 customers
    similar_customers = similarity_df[customer_id].sort_values(ascending=False)[1:4]  # Top 3, exclude self
    lookalike_results[customer_id] = list(zip(similar_customers.index, similar_customers.values))

# Save results to a CSV file
lookalike_df = pd.DataFrame({
    "CustomerID": lookalike_results.keys(),
    "Lookalikes": lookalike_results.values()
})
lookalike_df.to_csv('Lookalike.csv', index=False)

print("Lookalike model completed! Check the 'Lookalike.csv' file for results.")
