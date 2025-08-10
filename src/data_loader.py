
import pandas as pd
import numpy as np
import re

def load_and_initial_clean(file_path):
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: '{file_path}' not found. Please ensure it's in the correct directory.")
        return None

    # Drop duplicate rows
    initial_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    if df.shape[0] < initial_rows:
        print(f"Removed {initial_rows - df.shape[0]} duplicate rows.")

    new_columns = []
    for col in df.columns:
        # 1. Convert to lowercase
        cleaned_col = col.lower()
        # 2. Replace any character that is NOT a lowercase letter, number, or underscore with an underscore
        cleaned_col = re.sub(r'[^a-z0-9_]', '_', cleaned_col)
        # 3. Replace multiple underscores with a single underscore
        cleaned_col = re.sub(r'_{2,}', '_', cleaned_col)
        # 4. Strip leading/trailing underscores
        cleaned_col = cleaned_col.strip('_')
        new_columns.append(cleaned_col)
    df.columns = new_columns
    print("DataFrame columns standardized using a robust method in data_loader.py.")

    # Drop 'order' and 'pid' columns 
    if 'order' in df.columns:
        df.drop('order', axis=1, inplace=True)
        print("Dropped 'order' column.")
    if 'pid' in df.columns:
        df.drop('pid', axis=1, inplace=True)
        print("Dropped 'pid' column.")

    print(f"Data loaded and initially cleaned. Shape: {df.shape}")
    return df

if __name__ == '__main__':
    
    DATA_PATH = '../data/AmesHousing.csv'
    clean_df = load_and_initial_clean(DATA_PATH)
    if clean_df is not None:
        print("\nCleaned DataFrame Head:")
        print(clean_df.head())
        print("\nCleaned DataFrame Info:")
        clean_df.info()