"""
Run color analysis script on the labelled dataset and integrate results. 
"""

import os
from pathlib import Path
from color_analyzer import ColorAnalyzer
import logging
import pandas as pd
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_color_analysis(input_csv=None, output_csv=None, n_colors=5):
    """Run color analysis on fashion labels and integrate results.
    
    Args:
        input_csv: Path to input fashion_labels.csv
        output_csv: Path to save output CSV with color data
        n_colors: Number of dominant colors to extract per image
    """
    try:
        # Set up paths
        project_root = Path(__file__).parent.parent.parent
        
        # Default paths if not provided
        if input_csv is None:
            input_csv = project_root / 'data' / 'processed' / 'fashion_labels.csv'
        else:
            input_csv = Path(input_csv)
            
        if output_csv is None:
            output_csv = project_root / 'data' / 'processed' / 'fashion_labels_with_colors.csv'
        else:
            output_csv = Path(output_csv)
            
        # Ensure input CSV exists
        if not input_csv.exists():
            raise FileNotFoundError(f"Input CSV not found: {input_csv}")
            
        # Create output directory if needed
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Starting color analysis pipeline")
        logging.info(f"Input CSV: {input_csv}")
        logging.info(f"Output CSV: {output_csv}")
        
        # Initialize color analyzer
        analyzer = ColorAnalyzer(n_colors=n_colors)
        
        # Process fashion labels
        result_df = analyzer.process_fashion_labels(
            labels_path=str(input_csv),
            output_path=str(output_csv)
        )
        
        if result_df is not None:
            # Generate summary statistics
            color_stats = {}
            for i in range(1, n_colors + 1):
                color_col = f'color_{i}'
                pct_col = f'color_{i}_percentage'
                
                # Get top colors by frequency
                color_counts = result_df[color_col].value_counts().head(10)
                
                # Get average percentage for each color
                color_avg_pct = result_df.groupby(color_col)[pct_col].mean()
                
                color_stats[f'top_colors_{i}'] = {
                    color: {
                        'count': count,
                        'avg_percentage': color_avg_pct[color]
                    }
                    for color, count in color_counts.items()
                }
            
            # Save color statistics
            stats_file = output_csv.parent / 'color_statistics.json'
            import json
            with open(stats_file, 'w') as f:
                json.dump(color_stats, f, indent=2)
            
            logging.info(f"\nColor analysis complete!")
            logging.info(f"Updated fashion labels saved to: {output_csv}")
            logging.info(f"Color statistics saved to: {stats_file}")
            
            # Print some insights
            logging.info("\nQuick color insights:")
            for i in range(1, n_colors + 1):
                top_colors = list(color_stats[f'top_colors_{i}'].items())[:3]
                logging.info(f"\nTop 3 colors (position {i}):")
                for color, stats in top_colors:
                    logging.info(f"  {color}: {stats['count']} occurrences, {stats['avg_percentage']:.1f}% average coverage")
        
        else:
            logging.error("Failed to process fashion labels!")
            
    except Exception as e:
        logging.error(f"Error running color analysis: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Run color analysis on fashion labels')
    parser.add_argument('--input_csv', type=str, help='Path to input fashion_labels.csv')
    parser.add_argument('--output_csv', type=str, help='Path to save output CSV with color data')
    parser.add_argument('--n_colors', type=int, default=5, help='Number of dominant colors to extract')
    
    args = parser.parse_args()
    run_color_analysis(args.input_csv, args.output_csv, args.n_colors)

if __name__ == "__main__":
    main() 