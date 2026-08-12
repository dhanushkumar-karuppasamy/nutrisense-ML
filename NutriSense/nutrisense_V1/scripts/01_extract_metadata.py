import os
import pandas as pd
# pyrefly: ignore [missing-import]
import pyreadstat

def extract_metadata(dta_path, output_csv):
    print(f"Extracting metadata from {dta_path}...")
    if not os.path.exists(dta_path):
        print(f"File not found: {dta_path}")
        return
    
    # Read metadata only (much faster and memory efficient)
    _, metadata = pyreadstat.read_dta(dta_path, metadataonly=True)
    
    # Create DataFrame of variables and their labels
    df_meta = pd.DataFrame({
        'variable': metadata.column_names,
        'label': [metadata.column_names_to_labels.get(col, '') for col in metadata.column_names]
    })
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_meta.to_csv(output_csv, index=False)
    print(f"Saved metadata to {output_csv}")

if __name__ == "__main__":
    raw_dir = os.path.join("data", "raw")
    interim_dir = os.path.join("data", "interim")
    
    datasets = {
        "IAKR7EFL.DTA": "kr_metadata.csv",
        "IAHR7EFL.DTA": "hr_metadata.csv",
        "IAIR7EFL.DTA": "ir_metadata.csv"
    }
    
    for dta, csv_out in datasets.items():
        extract_metadata(
            os.path.join(raw_dir, dta),
            os.path.join(interim_dir, csv_out)
        )
