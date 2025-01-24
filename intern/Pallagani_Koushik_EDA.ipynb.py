import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
customers_df = pd.read_csv('Customers.csv')
products_df = pd.read_csv('Products.csv')
transactions_df = pd.read_csv('Transactions.csv')

# Convert date columns to datetime
customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'])
transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'])

# Summary statistics
print("Customers Summary:")
print(customers_df.describe(include='all'))
print("\nProducts Summary:")
print(products_df.describe(include='all'))
print("\nTransactions Summary:")
print(transactions_df.describe(include='all'))

# Distribution of Regions (Customers)
plt.figure(figsize=(8, 5))
sns.countplot(data=customers_df, x='Region', palette='viridis')
plt.title('Distribution of Customers by Region')
plt.xlabel('Region')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Distribution of Product Categories
plt.figure(figsize=(8, 5))
sns.countplot(data=products_df, x='Category', palette='plasma')
plt.title('Distribution of Products by Category')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Total Transactions Over Time
transactions_df['TransactionMonth'] = transactions_df['TransactionDate'].dt.to_period('M')
transactions_over_time = transactions_df['TransactionMonth'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
transactions_over_time.plot(kind='bar', color='teal')
plt.title('Transactions Over Time (Monthly)')
plt.xlabel('Month')
plt.ylabel('Number of Transactions')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
