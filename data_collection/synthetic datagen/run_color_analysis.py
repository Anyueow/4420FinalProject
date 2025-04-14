"""
Run color analysis on runway images and save results.
"""

import os
from pathlib import Path
from color_analyzer import ColorAnalyzer
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)

def run_color_analysis():
    """Run color analysis on runway images."""
    try:
        # Initialize color analyzer
        analyzer = ColorAnalyzer(n_colors=5)
        
        # Set up paths
        project_root = Path(__file__).parent.parent.parent
        runway_dir = project_root / 'data' / 'scraped' / 'runway'
        output_dir = project_root / 'data' / 'processed' / 'color_analysis'
        
        # Ensure runway directory exists
        if not runway_dir.exists():
            raise FileNotFoundError(f"Runway directory not found: {runway_dir}")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get season directories
        season_dirs = [d for d in runway_dir.iterdir() if d.is_dir()]
        if not season_dirs:
            raise ValueError(f"No season directories found in {runway_dir}")
        
        # Process each season
        all_designer_data = []
        all_season_data = []
        
        for season_dir in season_dirs:
            logging.info(f"\nProcessing season: {season_dir.name}")
            
            # Analyze season
            designer_data, season_data = analyzer.analyze_season(season_dir)
            
            if not designer_data.empty:
                all_designer_data.append(designer_data)
                all_season_data.append(season_data)
                logging.info(f"Successfully analyzed {season_dir.name}")
                
                # Plot season colors
                analyzer.plot_color_distribution(
                    season_data,
                    f"Color Distribution - {season_dir.name}"
                )
            else:
                logging.warning(f"No valid data found for {season_dir.name}")
        
        if all_designer_data and all_season_data:
            # Save combined results
            analyzer.save_results(
                pd.concat(all_designer_data, ignore_index=True),
                pd.concat(all_season_data, ignore_index=True),
                output_dir
            )
            
            logging.info("\nColor analysis complete!")
            logging.info(f"Results saved to: {output_dir}")
        else:
            logging.error("No valid data was processed!")
            
    except Exception as e:
        logging.error(f"Error running color analysis: {str(e)}")

if __name__ == "__main__":
    run_color_analysis() 