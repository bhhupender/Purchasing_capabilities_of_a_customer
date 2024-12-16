import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
file_path = "../data/final_cleaned_bank_transactions.csv"
data = pd.read_csv(file_path)

# Summary statistics for numerical columns
print("Summary Statistics:\n", data.describe())

# Inspect unique values for categorical columns
print("Unique Values in CustGender:\n", data["CustGender"].value_counts())
print("Unique Values in CustLocation:\n", data["CustLocation"].value_counts().head(10))  # Top 10 locations

# Age distribution
plt.figure(figsize=(10, 6))
sns.histplot(data["Age"], bins=20, kde=True, color="blue")
plt.title("Age Distribution of Customers")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Account Balance distribution
plt.figure(figsize=(10, 6))
sns.histplot(data["CustAccountBalance"], bins=50, kde=True, color="green")
plt.title("Account Balance Distribution")
plt.xlabel("Account Balance (INR)")
plt.ylabel("Frequency")
plt.show()

# Transaction Amount distribution
plt.figure(figsize=(10, 6))
sns.boxplot(x=data["TransactionAmount (INR)"])
plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount (INR)")
plt.show()

# Convert TransactionDate to datetime
data["TransactionDate"] = pd.to_datetime(data["TransactionDate"], errors='coerce')
data["Month"] = data["TransactionDate"].dt.to_period("M")

# Monthly transaction analysis
monthly_transactions = data.groupby("Month").size()

plt.figure(figsize=(12, 6))
monthly_transactions.plot(kind="line", marker="o", color="orange")
plt.title("Monthly Transactions")
plt.xlabel("Month")
plt.ylabel("Number of Transactions")
plt.grid(True)
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
sns.boxplot(x="CustGender", y="CustAccountBalance", data=data, palette="Set2")
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
