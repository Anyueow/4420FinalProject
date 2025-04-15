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
            # Try both original and color-enhanced filenames
            file_paths = [
                self.data_dir / f"fashion_labels_{season.lower()}_with_colors.csv",
                self.data_dir / f"fashion_labels_{season.lower()}.csv"
            ]
            
            loaded = False
            for file_path in file_paths:
                if file_path.exists():
                    self.data[season] = pd.read_csv(file_path)
                    print(f"Loaded data for {season} from {file_path}")
                    loaded = True
                    break
                    
            if not loaded:
                print(f"Warning: No data found for {season}")
    
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
    
    def aggregate_colors(self) -> Dict[str, Dict]:
        """Aggregate color frequencies and trends across seasons."""
        color_stats = {}
        
        for season, df in self.data.items():
            # Check if color columns exist
            color_columns = [col for col in df.columns if col.startswith('color_') and not col.endswith('_percentage')]
            if not color_columns:
                print(f"Warning: No color data found for {season}")
                continue
                
            season_colors = {}
            
            # Aggregate primary colors (color_1)
            primary_colors = df['color_1'].value_counts().to_dict()
            season_colors['primary_colors'] = primary_colors
            
            # Aggregate all colors with their percentages
            all_colors = {}
            for i in range(1, 6):  # Assuming up to 5 colors per image
                color_col = f'color_{i}'
                pct_col = f'color_{i}_percentage'
                
                if color_col in df.columns and pct_col in df.columns:
                    color_data = df.groupby(color_col)[pct_col].mean().to_dict()
                    for color, percentage in color_data.items():
                        if color and not pd.isna(color):
                            if color in all_colors:
                                all_colors[color]['count'] += len(df[df[color_col] == color])
                                all_colors[color]['avg_percentage'] = (all_colors[color]['avg_percentage'] + percentage) / 2
                            else:
                                all_colors[color] = {
                                    'count': len(df[df[color_col] == color]),
                                    'avg_percentage': percentage
                                }
            
            season_colors['all_colors'] = all_colors
            
            # Calculate color combinations
            if len(color_columns) >= 2:
                color_combos = df.groupby(['color_1', 'color_2']).size().reset_index(name='count')
                top_combos = color_combos.nlargest(10, 'count').to_dict('records')
                season_colors['top_color_combinations'] = top_combos
            
            color_stats[season] = season_colors
        
        return color_stats
    
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
    
    def calculate_color_trends(self) -> Dict[str, Dict]:
        """Calculate changes in color usage between seasons."""
        color_trends = {}
        
        # Get all unique colors across seasons
        all_colors = set()
        for season_data in self.data.values():
            if 'color_1' in season_data.columns:
                all_colors.update(season_data['color_1'].dropna().unique())
        
        # Calculate changes between consecutive seasons
        for i in range(len(self.seasons) - 1):
            current_season = self.seasons[i]
            next_season = self.seasons[i + 1]
            
            if current_season in self.data and next_season in self.data:
                current_df = self.data[current_season]
                next_df = self.data[next_season]
                
                if 'color_1' not in current_df.columns or 'color_1' not in next_df.columns:
                    continue
                
                # Calculate primary color changes
                current_colors = current_df['color_1'].value_counts(normalize=True)
                next_colors = next_df['color_1'].value_counts(normalize=True)
                
                # Calculate percentage changes
                color_changes = {}
                for color in all_colors:
                    current_pct = current_colors.get(color, 0)
                    next_pct = next_colors.get(color, 0)
                    change = next_pct - current_pct
                    if abs(change) > 0.01:  # Only include significant changes
                        color_changes[color] = {
                            'change': change,
                            'current_percentage': current_pct,
                            'next_percentage': next_pct
                        }
                
                color_trends[f"{current_season}_to_{next_season}"] = {
                    'color_changes': color_changes,
                    'emerging_colors': [c for c, v in color_changes.items() if v['change'] > 0.05],
                    'declining_colors': [c for c, v in color_changes.items() if v['change'] < -0.05]
                }
        
        return color_trends

    def analyze(self) -> Dict:
        """Run all analyses and return aggregated results."""
        self.load_data()
        
        if not self.data:
            raise ValueError("No data loaded. Please check the data directory and file paths.")
        
        results = {
            'categories': self.aggregate_categories(),
            'styles': self.aggregate_styles(),
            'patterns': self.aggregate_patterns(),
            'colors': self.aggregate_colors(),
            'trend_changes': self.calculate_trend_changes(),
            'color_trends': self.calculate_color_trends()
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
        
        # Print color trend highlights
        if 'color_trends' in results:
            print("\nColor Trend Highlights:")
            for transition, trends in results['color_trends'].items():
                print(f"\n{transition}:")
                if trends['emerging_colors']:
                    print("Emerging colors:", ", ".join(trends['emerging_colors']))
                if trends['declining_colors']:
                    print("Declining colors:", ", ".join(trends['declining_colors']))
                
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main() 