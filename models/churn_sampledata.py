import pandas as pd
import random

# Generate 1000 rows of sample data
random.seed(42)
data = {
    "Recency": [random.choice([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]) for _ in range(10000)],
    "Frequency": [random.randint(1, 6) for _ in range(10000)],
    "Monetary": [random.randint(100, 10000) for _ in range(10000)],
    "Cluster": [random.randint(0, 3) for _ in range(10000)],
    "Churn": [random.choice([0, 1]) for _ in range(10000)]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save as Excel file
df.to_csv("../data/churn_test_data.csv", index=False)
