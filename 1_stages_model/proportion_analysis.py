import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class MenopauseDeclineAnalysis:
    """
    Analyzes the proportion of women experiencing cognitive decline 
    or emotional worsening across different menopausal stages.
    """
    
    def __init__(self, data):
        self.data = data
        self.output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                       '1_stages_model', 'output', 'longitudinal')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define cognitive and emotional measures
        self.cognitive_measures = ['TOTIDE1', 'TOTIDE2', 'TOTIDE_avg']
        self.emotional_measures = ['NERVES', 'SAD', 'FEARFULA', 'NERVES_log', 'SAD_sqrt', 'FEARFULA_sqrt']
        
        # Status order for visualization
        self.status_order = ['Pre-menopause', 'Early Peri', 'Late Peri', 'Post-menopause', 'Surgical']
    
    def calculate_decline_proportions(self):
        """
        Calculate the proportion of women who experience decline in cognitive 
        and emotional measures across visits, grouped by menopausal stage.
        """
        # Ensure we have the STATUS_Label column
        if 'STATUS_Label' not in self.data.columns:
            print("Error: STATUS_Label column not found. Please ensure filter_status() has been run.")
            return None
        
        # Convert measures to numeric if needed
        for col in self.cognitive_measures + self.emotional_measures:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        # Group by woman ID and sort by visit to calculate changes
        decline_data = []
        
        # Process all subjects with multiple visits
        subjects = self.data['SWANID'].unique()
        for subject in subjects:
            subject_data = self.data[self.data['SWANID'] == subject].sort_values('VISIT')
            
            # Get the status label (using the most recent status)
            status = subject_data['STATUS_Label'].iloc[-1]
            
            # Calculate changes in measures
            for visit_idx in range(1, len(subject_data)):
                prev_visit = subject_data.iloc[visit_idx-1]
                curr_visit = subject_data.iloc[visit_idx]
                
                # Cognitive measures (higher is better, so decline is negative change)
                for measure in [m for m in self.cognitive_measures if m in self.data.columns]:
                    if not pd.isna(prev_visit[measure]) and not pd.isna(curr_visit[measure]):
                        change = curr_visit[measure] - prev_visit[measure]
                        decline_data.append({
                            'SWANID': subject,
                            'STATUS_Label': status,
                            'Measure': measure,
                            'Category': 'Cognitive',
                            'Change': change,
                            'Has_Decline': 1 if change < 0 else 0
                        })
                
                # Emotional measures (higher is worse, so worsening is positive change)
                for measure in [m for m in self.emotional_measures if m in self.data.columns]:
                    if not pd.isna(prev_visit[measure]) and not pd.isna(curr_visit[measure]):
                        change = curr_visit[measure] - prev_visit[measure]
                        decline_data.append({
                            'SWANID': subject,
                            'STATUS_Label': status,
                            'Measure': measure,
                            'Category': 'Emotional',
                            'Change': change,
                            'Has_Decline': 1 if change > 0 else 0
                        })
        
        # Convert to DataFrame
        decline_df = pd.DataFrame(decline_data)
        
        # Filter to main measures to avoid duplicates from transformed variables
        main_measures = ['TOTIDE_avg', 'NERVES', 'SAD', 'FEARFULA']
        filtered_decline_df = decline_df[decline_df['Measure'].isin(main_measures)]
        
        # Calculate proportion with decline by status and measure
        proportions = filtered_decline_df.groupby(['STATUS_Label', 'Measure', 'Category'])['Has_Decline'].mean()
        proportions_df = proportions.reset_index()
        
        # Calculate overall proportions by status and category
        category_proportions = filtered_decline_df.groupby(['STATUS_Label', 'Category'])['Has_Decline'].mean()
        category_proportions_df = category_proportions.reset_index()
        
        return proportions_df, category_proportions_df
    
    def plot_decline_proportions(self, proportions_df):
        """
        Create a visualization showing the proportion of women experiencing
        decline across all four measures (cognitive, nervousness, sadness, fearfulness)
        by menopausal stage.
        """
        if proportions_df is None or proportions_df.empty:
            print("Error: No decline proportion data available.")
            return
        
        # Ensure STATUS_Label is in the correct order
        if 'STATUS_Label' in proportions_df.columns:
            proportions_df['STATUS_Label'] = pd.Categorical(
                proportions_df['STATUS_Label'],
                categories=self.status_order,
                ordered=True
            )
            proportions_df = proportions_df.sort_values('STATUS_Label')
        
        # Define measure order and labels
        measure_order = ['TOTIDE_avg', 'NERVES', 'SAD', 'FEARFULA']
        measure_labels = {
            'TOTIDE_avg': 'Cognitive Function',
            'NERVES': 'Nervousness', 
            'SAD': 'Sadness',
            'FEARFULA': 'Fearfulness'
        }
        
        # Filter to main measures
        plot_data = proportions_df[proportions_df['Measure'].isin(measure_order)].copy()
        
        # Create measure labels
        plot_data['Measure_Label'] = plot_data['Measure'].map(measure_labels)
        
        # Initialize the plot with subplots for each measure
        fig, axes = plt.subplots(4, 1, figsize=(14, 20))
        axes = axes.flatten()
        
        # Use the YlGn color palette 
        green_palette = sns.color_palette("YlGn", n_colors=8)
        colors = [green_palette[3], green_palette[4], green_palette[5], green_palette[6]]
        
        for idx, measure in enumerate(measure_order):
            ax = axes[idx]
            
            # Filter data for this measure
            measure_data = plot_data[plot_data['Measure'] == measure]
            
            if measure_data.empty:
                continue
                
            # Create bar plot
            bars = ax.bar(
                range(len(measure_data)), 
                measure_data['Has_Decline'],
                color=colors[idx],
                alpha=0.8,
                edgecolor='black',
                linewidth=0.5
            )
            
            # Add value labels on top of each bar
            for i, (bar, value) in enumerate(zip(bars, measure_data['Has_Decline'])):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2, 
                    height + 0.01,
                    f'{value*100:.1f}%', 
                    ha='center',
                    va='bottom',
                    fontweight='bold',
                    fontsize=18
                )
            
            # Customize subplot
            ax.set_title(f'{measure_labels[measure]}', fontsize=18, fontweight='bold')
            
            # Set x-axis labels
            ax.set_xticks(range(len(measure_data)))
            ax.set_xticklabels(measure_data['STATUS_Label'], rotation=45, ha='right', fontsize=18)
            ax.tick_params(axis='y', labelsize=18)
            
            # Format y-axis as percentage
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
            
            # Set consistent y-axis limits
            ax.set_ylim(0, max(0.3, measure_data['Has_Decline'].max() * 1.2))
            
            # Add grid for better readability
            ax.grid(True, axis='y', linestyle=':', alpha=0.6)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # Save the figure
        output_path = os.path.join(self.output_dir, 'menopausal_decline_proportions.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved to {output_path}")
        
        return fig
    
    def tabulate_results(self, proportions_df, category_proportions_df):
        """
        Create tables showing the detailed results for reporting.
        """
        # Format proportions as percentages
        proportions_df['Decline_Percentage'] = (proportions_df['Has_Decline'] * 100).round(1)
        category_proportions_df['Decline_Percentage'] = (category_proportions_df['Has_Decline'] * 100).round(1)
        
        # Count number of women in each stage who had multiple visits
        women_counts = self.data.groupby('STATUS_Label')['SWANID'].nunique()
        
        # Print detailed results
        print("\nProportion of Women Experiencing Decline by Menopausal Stage and Measure Type:")
        print("=" * 80)
        print(f"{'Status':<15} {'Category':<10} {'Proportion (%)':<15} {'Sample Size':<12}")
        print("-" * 80)
        
        for status in self.status_order:
            if status in category_proportions_df['STATUS_Label'].values:
                for category in ['Cognitive', 'Emotional']:
                    row = category_proportions_df[
                        (category_proportions_df['STATUS_Label'] == status) & 
                        (category_proportions_df['Category'] == category)
                    ]
                    if not row.empty:
                        sample_size = women_counts.get(status, 0)
                        print(f"{status:<15} {category:<10} {row['Decline_Percentage'].values[0]:<15.1f} {sample_size:<12}")
        
        print("=" * 80)
        print("\nDetailed Breakdown by Specific Measures:")
        print("=" * 80)
        print(f"{'Status':<15} {'Measure':<10} {'Category':<10} {'Proportion (%)':<15}")
        print("-" * 80)
        
        for status in self.status_order:
            status_rows = proportions_df[proportions_df['STATUS_Label'] == status]
            if not status_rows.empty:
                for _, row in status_rows.iterrows():
                    print(f"{status:<15} {row['Measure']:<10} {row['Category']:<10} {row['Decline_Percentage']:<15.1f}")
        
        print("=" * 80)
        
        # Return formatted dataframes for further analysis
        return proportions_df, category_proportions_df
    
    def run_analysis(self):
        """
        Run the complete decline proportion analysis.
        """
        print("\nAnalyzing proportion of women experiencing cognitive and emotional decline...")
        
        # Calculate decline proportions
        proportions_df, category_proportions_df = self.calculate_decline_proportions()
        
        if proportions_df is not None and category_proportions_df is not None:
            # Create visualization
            self.plot_decline_proportions(proportions_df)
            
            # Generate detailed tables
            formatted_prop_df, formatted_cat_df = self.tabulate_results(proportions_df, category_proportions_df)
            
            return formatted_prop_df, formatted_cat_df
        else:
            print("Error: Could not calculate decline proportions.")
            return None, None