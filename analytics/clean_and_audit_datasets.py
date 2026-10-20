import os
import glob
import pandas as pd

def clean_and_audit():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    csv_files = sorted(glob.glob(os.path.join(data_dir, "[0-9][0-9]_*.csv")))
    
    if not csv_files:
        print("No raw dataset CSV files found in data directory.")
        return

    for filepath in csv_files:
        filename = os.path.basename(filepath)
        df = pd.read_csv(filepath)
        
        # Forward and backward fill missing values
        df_cleaned = df.ffill().bfill()
        
        output_filename = f"cleaned_{filename}"
        output_path = os.path.join(data_dir, output_filename)
        df_cleaned.to_csv(output_path, index=False)
        print(f"Audited and cleaned: {filename} -> {output_filename}")

if __name__ == "__main__":
    clean_and_audit()