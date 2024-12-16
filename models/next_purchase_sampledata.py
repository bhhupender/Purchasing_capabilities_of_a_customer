import pandas as pd
import random

# Generate 10000 rows of sample data
random.seed(42)
data = {
    "Recency": [random.choice([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]) for _ in range(10000)],
    "Frequency": [random.randint(1, 6) for _ in range(10000)],
    "Monetary": [random.randint(100, 10000) for _ in range(10000)],
    "Cluster": [random.randint(0, 3) for _ in range(10000)],
    "TimeToNextPurchase": [random.randint(1, 360) for _ in range(10000)]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save as CSV file
file_name = "../data/test_data_next_purchase.csv"
df.to_csv(file_name, index=False)

print(f"Sample test data saved as {file_name}")
