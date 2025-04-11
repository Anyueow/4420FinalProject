"""
Visualize fashion trend analysis results.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict
import os

class TrendVisualizer:
    """Visualize fashion trend analysis results."""
    
    def __init__(self, trends_data: Dict, save_dir: str = "results/visualizations"):
        self.trends_data = trends_data
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn')
        sns.set_palette("husl")
    
    def plot_top_trends(self, n: int = 10):
        """Plot top N trends by total appearances."""
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'trend': trend,
                'appearances': data['total_appearances'],
                'designers': data['unique_designer_count']
            }
            for trend, data in self.trends_data.items()
        ])
        
        # Sort and get top N
        df = df.nlargest(n, 'appearances')
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot bars
        bars = plt.bar(df['trend'], df['appearances'])
        
        # Customize plot
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Top {n} Fashion Trends by Appearances')
        plt.xlabel('Trend')
        plt.ylabel('Number of Appearances')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'top_trends.png'))
        plt.close()
    
    def plot_designer_distribution(self, n: int = 10):
        """Plot designer distribution for top N trends."""
        # Convert to DataFrame
        data = []
        for trend, trend_data in self.trends_data.items():
            for designer in trend_data['designer_appearances']:
                data.append({
                    'trend': trend,
                    'designer': designer,
                    'appearances': trend_data['shows'][designer]
                })
        
        df = pd.DataFrame(data)
        
        # Get top N trends
        top_trends = df.groupby('trend')['appearances'].sum().nlargest(n).index
        df_top = df[df['trend'].isin(top_trends)]
        
        # Create heatmap
        pivot_table = df_top.pivot_table(
            values='appearances',
            index='trend',
            columns='designer',
            fill_value=0
        )
        
        # Plot heatmap
        plt.figure(figsize=(15, 8))
        sns.heatmap(pivot_table, annot=True, fmt='d', cmap='YlOrRd')
        
        plt.title(f'Designer Distribution for Top {n} Trends')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'designer_distribution.png'))
        plt.close()
    
    def plot_trend_timeline(self, n: int = 5):
        """Plot timeline of top N trends across shows."""
        # Convert to DataFrame with show order
        data = []
        for trend, trend_data in self.trends_data.items():
            for designer, count in trend_data['shows'].items():
                data.append({
                    'trend': trend,
                    'designer': designer,
                    'appearances': count
                })
        
        df = pd.DataFrame(data)
        
        # Get top N trends
        top_trends = df.groupby('trend')['appearances'].sum().nlargest(n).index
        df_top = df[df['trend'].isin(top_trends)]
        
        # Create line plot
        plt.figure(figsize=(15, 8))
        
        for trend in top_trends:
            trend_data = df_top[df_top['trend'] == trend]
            plt.plot(trend_data['designer'], trend_data['appearances'],
                    marker='o', label=trend)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Trend Timeline Across Shows')
        plt.xlabel('Designer Shows (in order)')
        plt.ylabel('Number of Appearances')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'trend_timeline.png'))
        plt.close()
    
    def generate_report(self):
        """Generate a comprehensive trend report."""
        report = {
            'summary': {
                'total_trends': len(self.trends_data),
                'total_designers': len(set(
                    designer
                    for trend_data in self.trends_data.values()
                    for designer in trend_data['designer_appearances']
                )),
            },
            'top_trends': sorted(
                [
                    {
                        'trend': trend,
                        'total_appearances': data['total_appearances'],
                        'unique_designers': data['unique_designer_count'],
                        'top_designers': sorted(
                            data['shows'].items(),
                            key=lambda x: x[1],
                            reverse=True
                        )[:5]
                    }
                    for trend, data in self.trends_data.items()
                ],
                key=lambda x: x['total_appearances'],
                reverse=True
            )[:10]
        }
        
        # Save report
        with open(os.path.join(self.save_dir, 'trend_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        return report

def visualize_trends(trends_data: Dict, save_dir: str = "results/visualizations"):
    """Main function to create all visualizations."""
    visualizer = TrendVisualizer(trends_data, save_dir)
    
    # Generate all plots
    visualizer.plot_top_trends()
    visualizer.plot_designer_distribution()
    visualizer.plot_trend_timeline()
    
    # Generate report
    report = visualizer.generate_report()
    
    return report 