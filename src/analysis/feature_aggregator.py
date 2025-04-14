import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
import os

class FashionFeatureAggregator:
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            # Get the project root directory
            project_root = Path(__file__).parent.parent.parent
            self.data_dir = project_root / "data" / "processed"
        else:
            self.data_dir = Path(data_dir)
        
        self.seasons = ["Fall24", "Fall25", 'Spring25']  # Add more seasons as needed
        self.data = {}
        self.aggregated_features = {}
        
    def load_data(self) -> None:
        """Load data from all season CSV files."""
        for season in self.seasons:
            file_path = self.data_dir / f"fashion_labels_{season.lower()}.csv"
            if file_path.exists():
                self.data[season] = pd.read_csv(file_path)
                print(f"Loaded data for {season}")
            else:
                print(f"Warning: No data found for {season} at {file_path}")
    
    def aggregate_categories(self) -> Dict[str, Dict]:
        """Aggregate category frequencies across seasons."""
        category_stats = {}
        for season, df in self.data.items():
            # Count occurrences of each category
            category_counts = df['category'].value_counts().to_dict()
            super_category_counts = df['super_category'].value_counts().to_dict()
            
            category_stats[season] = {
                'categories': category_counts,
                'super_categories': super_category_counts,
                'total_items': len(df)
            }
        return category_stats
    
    def aggregate_styles(self) -> Dict[str, Dict]:
        """Aggregate style frequencies across seasons."""
        style_stats = {}
        for season, df in self.data.items():
            # Filter out empty style entries
            style_df = df[df['style'].notna() & (df['style'] != '')]
            super_style_df = df[df['super_style'].notna() & (df['super_style'] != '')]
            
            style_counts = style_df['style'].value_counts().to_dict()
            super_style_counts = super_style_df['super_style'].value_counts().to_dict()
            
            style_stats[season] = {
                'styles': style_counts,
                'super_styles': super_style_counts,
                'total_styled_items': len(style_df)
            }
        return style_stats
    
    def aggregate_patterns(self) -> Dict[str, Dict]:
        """Aggregate pattern frequencies across seasons."""
        pattern_stats = {}
        for season, df in self.data.items():
            # Filter out empty pattern entries
            pattern_df = df[df['pattern'].notna() & (df['pattern'] != '')]
            super_pattern_df = df[df['super_pattern'].notna() & (df['super_pattern'] != '')]
            
            pattern_counts = pattern_df['pattern'].value_counts().to_dict()
            super_pattern_counts = super_pattern_df['super_pattern'].value_counts().to_dict()
            
            pattern_stats[season] = {
                'patterns': pattern_counts,
                'super_patterns': super_pattern_counts,
                'total_patterned_items': len(pattern_df)
            }
        return pattern_stats
    
    def calculate_trend_changes(self) -> Dict[str, Dict]:
        """Calculate changes in feature frequencies between seasons."""
        trend_changes = {}
        
        # Get all unique features across seasons
        all_categories = set()
        all_styles = set()
        all_patterns = set()
        
        for season_data in self.data.values():
            all_categories.update(season_data['category'].unique())
            all_styles.update(season_data['style'].dropna().unique())
            all_patterns.update(season_data['pattern'].dropna().unique())
        
        # Calculate changes between consecutive seasons
        for i in range(len(self.seasons) - 1):
            current_season = self.seasons[i]
            next_season = self.seasons[i + 1]
            
            if current_season in self.data and next_season in self.data:
                current_df = self.data[current_season]
                next_df = self.data[next_season]
                
                # Calculate category changes
                current_cats = current_df['category'].value_counts(normalize=True)
                next_cats = next_df['category'].value_counts(normalize=True)
                
                # Calculate style changes
                current_styles = current_df['style'].value_counts(normalize=True)
                next_styles = next_df['style'].value_counts(normalize=True)
                
                # Calculate pattern changes
                current_patterns = current_df['pattern'].value_counts(normalize=True)
                next_patterns = next_df['pattern'].value_counts(normalize=True)
                
                trend_changes[f"{current_season}_to_{next_season}"] = {
                    'category_changes': (next_cats - current_cats).to_dict(),
                    'style_changes': (next_styles - current_styles).to_dict(),
                    'pattern_changes': (next_patterns - current_patterns).to_dict()
                }
        
        return trend_changes
    
    def analyze(self) -> Dict:
        """Run all analyses and return aggregated results."""
        self.load_data()
        
        if not self.data:
            raise ValueError("No data loaded. Please check the data directory and file paths.")
        
        results = {
            'categories': self.aggregate_categories(),
            'styles': self.aggregate_styles(),
            'patterns': self.aggregate_patterns(),
            'trend_changes': self.calculate_trend_changes()
        }
        
        # Ensure output directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Save results to JSON
        output_path = self.data_dir / "feature_analysis.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

def main():
    try:
        aggregator = FashionFeatureAggregator()
        results = aggregator.analyze()
        print("Feature analysis completed. Results saved to feature_analysis.json")
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main() 