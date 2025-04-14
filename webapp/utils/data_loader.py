import pandas as pd
from pathlib import Path
import json

def load_fashion_data():
    """
    Load and process fashion data for the Streamlit app.
    Returns a pandas DataFrame with processed fashion data.
    """
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent
    
    # Load fashion labels
    labels_path = project_root / "data" / "fashion_labels.csv"
    df = pd.read_csv(labels_path)
    
    # Extract designer name from image path
    df['designer'] = df['image_path'].apply(lambda x: x.split('/')[-2])
    
    # Load color dictionary if available
    color_dict_path = project_root / "data" / "scraped" / "runway" / "Fall25" / "color_dictionary.json"
    if color_dict_path.exists():
        with open(color_dict_path, 'r') as f:
            color_data = json.load(f)
        # Add color data to DataFrame
        # This will be implemented based on the color analysis structure
    
    return df

def load_designer_data(designer_name):
    """
    Load specific data for a designer.
    """
    df = load_fashion_data()
    return df[df['designer'] == designer_name]

def load_color_data():
    """
    Load color analysis data.
    """
    project_root = Path(__file__).parent.parent.parent
    color_dict_path = project_root / "data" / "scraped" / "runway" / "Fall25" / "color_dictionary.json"
    
    if color_dict_path.exists():
        with open(color_dict_path, 'r') as f:
            return json.load(f)
    return None
