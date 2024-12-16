import pandas as pd
from datetime import datetime

# Load the dataset
def load_data(file_path):
    """Load the dataset from a CSV file."""
    data = pd.read_csv(file_path)
    return data

# Handle Missing Values
def clean_data(df):
    """Handle missing values in the dataset."""
    
    #print(df["CustomerDOB"].isnull().sum())
    # Drop rows with missing CustomerDOB since age is critical for segmentation
    df = df.dropna(subset=["CustomerDOB"]).copy()
    
    # Fill missing gender with 'M'
    df.loc[:, "CustGender"] = df["CustGender"].fillna("M")
    
    # Fill missing location with 'Mumbai'
    df.loc[:, "CustLocation"] = df["CustLocation"].fillna("Mumbai")
    
    # Fill missing account balance with the median balance
    df.loc[:, "CustAccountBalance"] = df["CustAccountBalance"].fillna(df["CustAccountBalance"].median())
    
    return df

# Handle date parsing errors
def safe_parse_date(dob):
    """Parse date safely, handling multiple delimiters and formats."""
    try:
        # Try standard dd/mm/yy format with "/"
        return datetime.strptime(dob, "%d/%m/%y")
    except ValueError:
        try:
            # Try full year format with "/"
            return datetime.strptime(dob, "%d/%m/%Y")
        except ValueError:
            try:
                # Try standard format with "-" delimiter
                return datetime.strptime(dob, "%d-%m-%Y")
            except ValueError:
                try:
                    # Try short year format with "-" delimiter
                    return datetime.strptime(dob, "%d-%m-%y")
                except ValueError:
                    return None  # Return None for invalid formats

# Calculate Age
def calculate_age_safe(dob):
    """Calculate age, accounting for None values."""
    if dob is None:
        return None
    today = datetime.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

# Convert CustomerDOB to Age
def add_cleaned_age_column(df):
    """Add a cleaned 'Age' column."""
    df = df.copy()  # Ensure a copy to avoid chained assignment
    df["ParsedDOB"] = df["CustomerDOB"].apply(safe_parse_date)  # Safely parse dates
    print("Invalid DOB rows after update:", df[df["ParsedDOB"].isnull()])
    print("Total valid rows:", len(df[df["ParsedDOB"].notnull()]))

    df["Age"] = df["ParsedDOB"].apply(calculate_age_safe)
    df.drop(columns=["ParsedDOB"], inplace=True)
    return df


# Remove rows with invalid (negative) ages
def remove_invalid_ages(df):
    """Remove rows with invalid (negative) ages."""
    return df[df["Age"] >= 0]

# Standardize Date and Time
def standardize_datetime(df):
    """Standardize date and time columns."""
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], format="%d/%m/%y")
    df["TransactionTime"] = df["TransactionTime"].apply(lambda x: f"{x:06d}")  # Ensure HHMMSS format
    df["TransactionTime"] = pd.to_datetime(df["TransactionTime"], format="%H%M%S").dt.time
    return df

# Main execution
def main():
    file_path = "../data/bank_transactions.csv"  # Path to dataset
    data = load_data(file_path)

    # Cleaning and transforming the data
    cleaned_data = clean_data(data)
    cleaned_data = add_cleaned_age_column(cleaned_data)
    #cleaned_data = remove_invalid_ages(cleaned_data)  # Remove invalid ages
    #cleaned_data = standardize_datetime(cleaned_data)  # Optionally standardize date and time

    # Save the cleaned data
    cleaned_file_path = "../data/final_cleaned_bank_transactions.csv"
    cleaned_data.to_csv(cleaned_file_path, index=False)

    # Display the final dataset information
    print(cleaned_data.info())
    print(cleaned_data.head())

if __name__ == "__main__":
    main()
