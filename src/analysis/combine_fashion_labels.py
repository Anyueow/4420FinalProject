import pandas as pd
from pathlib import Path
import os

def combine_fashion_labels(data_dir: str = None):
    """Combine fashion label CSV files from different seasons into one file."""
    if data_dir is None:
        # Get the project root directory
        project_root = Path(__file__).parent.parent.parent
        data_dir = project_root / "data" / "processed"
    else:
        data_dir = Path(data_dir)
    
    # List all fashion label CSV files
    csv_files = list(data_dir.glob("fashion_labels_*.csv"))
    
    if not csv_files:
        raise ValueError("No fashion label CSV files found in the data directory")
    
    # Read and combine all CSV files
    combined_df = pd.DataFrame()
    
    for file in csv_files:
        season = file.stem.split('_')[-1]  # Extract season from filename
        df = pd.read_csv(file)
        df['season'] = season  # Add season column
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    
    # Save combined data
    output_path = data_dir / "fashion_labels.csv"
    combined_df.to_csv(output_path, index=False)
    print(f"Combined data saved to {output_path}")
    
    return combined_df

if __name__ == "__main__":
    try:
        df = combine_fashion_labels()
        print(f"Successfully combined {len(df)} records from {len(list(Path('data/processed').glob('fashion_labels_*.csv')))} files")
    except Exception as e:
        print(f"Error combining files: {e}") 