import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load the cleaned dataset
file_path = "../data/final_cleaned_bank_transactions.csv"
data = pd.read_csv(file_path)

# Summary statistics for numerical columns
print("Summary Statistics:\n", data.describe())

# Inspect unique values for categorical columns
print("Unique Values in CustGender:\n", data["CustGender"].value_counts())
print("Unique Values in CustLocation:\n", data["CustLocation"].value_counts().head(10))  # Top 10 locations

# Calculate RFM: Recency, Frequency, Monetary

# Ensure the TransactionDate is in datetime format
data["TransactionDate"] = pd.to_datetime(data["TransactionDate"], errors='coerce')

# Recency: Days since last transaction
last_transaction = data.groupby("CustomerID")["TransactionDate"].max()
last_transaction = (data["TransactionDate"].max() - last_transaction).dt.days
last_transaction = pd.DataFrame(last_transaction)
last_transaction.columns = ["Recency"]

# Frequency: Number of transactions per customer
frequency = data.groupby("CustomerID")["TransactionID"].count()
frequency = pd.DataFrame(frequency)
frequency.columns = ["Frequency"]

# Monetary: Total amount spent by each customer
monetary = data.groupby("CustomerID")["TransactionAmount (INR)"].sum()
monetary = pd.DataFrame(monetary)
monetary.columns = ["Monetary"]

# Combine RFM into a single DataFrame
rfm = pd.concat([last_transaction, frequency, monetary], axis=1)
# Add CustomerID to RFM DataFrame to make it accessible
rfm['CustomerID'] = rfm.index
# Impute missing values with median
rfm[['Recency', 'Frequency', 'Monetary']] = rfm[['Recency', 'Frequency', 'Monetary']].fillna(rfm[['Recency', 'Frequency', 'Monetary']].median())
print("\nRFM Sample:\n", rfm.head())

# Visualizing RFM Metrics

# Recency distribution
plt.figure(figsize=(10, 6))
sns.histplot(rfm["Recency"], bins=20, kde=True, color="purple")
plt.title("Recency Distribution (Days since Last Transaction)")
plt.xlabel("Days")
plt.ylabel("Frequency")
plt.show()

# Frequency distribution
plt.figure(figsize=(10, 6))
sns.histplot(rfm["Frequency"], bins=20, kde=True, color="orange")
plt.title("Frequency Distribution (Number of Transactions)")
plt.xlabel("Frequency")
plt.ylabel("Frequency")
plt.show()

# Monetary distribution
plt.figure(figsize=(10, 6))
sns.histplot(rfm["Monetary"], bins=50, kde=True, color="green")
plt.title("Monetary Distribution (Total Transaction Amount)")
plt.xlabel("Monetary (INR)")
plt.ylabel("Frequency")
plt.show()

# Correlation Analysis
# Calculate correlations between important features
correlation_matrix = data[["CustAccountBalance", "TransactionAmount (INR)", "Age"]].corr()

# Visualize the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# Gender distribution
gender_counts = data["CustGender"].value_counts()

plt.figure(figsize=(8, 6))
gender_counts.plot(kind="pie", autopct="%1.1f%%", startangle=140, colors=["skyblue", "lightcoral"])
plt.title("Gender Distribution of Customers")
plt.ylabel("")  # Hide the y-axis label
plt.show()

# Gender vs Account Balance
plt.figure(figsize=(10, 6))
sns.boxplot(x="CustGender", y="CustAccountBalance", data=data, hue="CustGender", palette="Set2", showfliers=False)
plt.title("Gender vs. Account Balance")
plt.xlabel("Gender")
plt.ylabel("Account Balance (INR)")
plt.show()

# Top locations by transaction count
top_locations = data["CustLocation"].value_counts().head(10)

plt.figure(figsize=(12, 6))
top_locations.plot(kind="bar", color="purple")
plt.title("Top 10 Locations by Transaction Count")
plt.xlabel("Location")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=45)
plt.show()

# Customer Segmentation (K-Means Clustering)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Normalize the data before clustering
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# Apply K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

# Create a summary report
report = """
Customer Segmentation and RFM Analysis Report
-------------------------------------------

Date of Report Generation: {date}

Summary of the Cleaned Dataset:
--------------------------------
- Total Customers: {total_customers}
- Total Transactions: {total_transactions}

Key Insights:
-------------
- Recency (Days since last transaction): 
    - Minimum: {recency_min}
    - Maximum: {recency_max}
    - Mean: {recency_mean}
    
- Frequency (Number of transactions): 
    - Minimum: {frequency_min}
    - Maximum: {frequency_max}
    - Mean: {frequency_mean}
    
- Monetary (Total transaction amount):
    - Minimum: {monetary_min}
    - Maximum: {monetary_max}
    - Mean: {monetary_mean}

RFM Segmentation Summary:
-------------------------
- Number of Customers in Each Cluster:
    - Cluster 0: {cluster_0_count}
    - Cluster 1: {cluster_1_count}
    - Cluster 2: {cluster_2_count}
    - Cluster 3: {cluster_3_count}

Visual Insights:
----------------
""".format(
    date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    total_customers=rfm['CustomerID'].nunique(),
    total_transactions=data['TransactionID'].nunique(),
    recency_min=rfm['Recency'].min(),
    recency_max=rfm['Recency'].max(),
    recency_mean=rfm['Recency'].mean(),
    frequency_min=rfm['Frequency'].min(),
    frequency_max=rfm['Frequency'].max(),
    frequency_mean=rfm['Frequency'].mean(),
    monetary_min=rfm['Monetary'].min(),
    monetary_max=rfm['Monetary'].max(),
    monetary_mean=rfm['Monetary'].mean(),
    cluster_0_count=rfm[rfm['Cluster'] == 0].shape[0],
    cluster_1_count=rfm[rfm['Cluster'] == 1].shape[0],
    cluster_2_count=rfm[rfm['Cluster'] == 2].shape[0],
    cluster_3_count=rfm[rfm['Cluster'] == 3].shape[0]
)

# Save the report to a text file
with open("../data/reports/customer_report.txt", "w") as f:
    f.write(report)

print("Report generated and saved as 'customer_report.txt'")

# Save the combined final data (including CustomerID, Recency, Frequency, Monetary, and Cluster)
#final_data = pd.concat([data[['CustomerID', 'TransactionID', 'TransactionDate', 'TransactionAmount (INR)']], rfm[['Recency', 'Frequency', 'Monetary', 'Cluster']]], axis=1)

# Reset the index of the rfm DataFrame to avoid the ambiguity
# Check if CustomerID is in the index and reset it if necessary
if 'CustomerID' in rfm.index.names:
    rfm = rfm.reset_index(drop=True)  # Reset index and avoid inserting CustomerID again

final_data = pd.merge(data, rfm, on="CustomerID", how="left")

# Save it to a CSV file
final_data.to_csv('../data/reports/final_customer_segmented_data.csv', index=False)
print("Final customer data with RFM and clusters saved as 'final_customer_segmented_data.csv'")

# Plot the clusters based on Recency and Frequency
plt.figure(figsize=(10, 6))
sns.scatterplot(x=rfm["Recency"], y=rfm["Frequency"], hue=rfm["Cluster"], palette="Set2", s=100, alpha=0.7)
plt.title("Customer Segmentation based on Recency and Frequency")
plt.xlabel("Recency (Days since Last Transaction)")
plt.ylabel("Frequency (Number of Transactions)")
plt.legend().set_visible(False)  # Disable the legend
plt.show()
