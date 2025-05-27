import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.save_output import get_output_dir

class MenopauseVisualisations:
    def __init__(self, data):
        self.data = data
        self.outcome_vars = ['TOTIDE1', 'TOTIDE2', 'NERVES', 'SAD', 'FEARFULA']
        self.transformed_vars = ['TOTIDE_avg', 'NERVES_log', 'SAD_sqrt', 'FEARFULA_sqrt']
        self.var_labels = {
            'TOTIDE_avg': 'Cognitive Performance',
            'NERVES_log': 'Nervousness (Log)',
            'SAD_sqrt': 'Sadness (Sqrt)',
            'FEARFULA_sqrt': 'Fearfulness (Sqrt)'
        }
        self.output_dir = get_output_dir('1_stages_model', 'longitudinal')
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_violin_plots(self):
        """
        Create violin plots for each symptom and outcome measure across menopausal stages.
        Uses orange-scale themed colors and improves data point visualization.
        """
        # Create a figure with subplots for outcome variables
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.flatten()
        
        # Define a green color palette
        green_palette = sns.color_palette("YlGn", n_colors=len(self.data['STATUS_Label'].cat.categories))
        
        # Plot each outcome measure
        for i, measure in enumerate(self.transformed_vars):
            ax = axes[i]
            
            # Create violin plot with boxplot inside
            sns.violinplot(
                data=self.data,
                x='STATUS_Label',
                y=measure,
                hue='STATUS_Label',  # Set hue to match x
                legend=False,        # Don't show redundant legend
                ax=ax,
                inner='box',         # Show boxplot inside violin
                palette=green_palette,  # Use green scale palette
            )
            
            # Add jittered stripplot with very small marker size and high transparency
            sns.stripplot(
                data=self.data.sample(min(500, len(self.data))),  # Use smaller sample
                x='STATUS_Label',
                y=measure,
                ax=ax,
                color='black',
                alpha=0.05,           # Very high transparency
                size=1,               # Very small points
                jitter=True,          # Add jitter to avoid overcrowding
                dodge=False           # Don't dodge points
            )
            
            # Set title and labels
            measure_name = self.var_labels.get(measure, measure)
            ax.set_title(f'Distribution of {measure_name}', fontsize=12)
            ax.set_xlabel('Menopausal Stage', fontsize=10)
            ax.set_ylabel(measure_name, fontsize=10)
            
            # Rotate x-tick labels for better visibility
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Add summary statistics as text
            # Calculate means and medians for each group
            means_by_stage = self.data.groupby('STATUS_Label', observed=True)[measure].mean().round(3)
            medians_by_stage = self.data.groupby('STATUS_Label', observed=True)[measure].median().round(3)
            
            # Calculate sample sizes for each group
            counts_by_stage = self.data.groupby('STATUS_Label', observed=True)[measure].count()
            
            # Add annotation in upper right corner
            stats_text = "Group Statistics:\n"
            for stage in self.data['STATUS_Label'].cat.categories:
                if stage in means_by_stage:
                    mean_val = means_by_stage[stage]
                    median_val = medians_by_stage[stage]
                    count = counts_by_stage[stage]
                    stats_text += f"{stage} (n={count}):\n"
                    stats_text += f"  μ={mean_val}, Md={median_val}\n"
                
            # Place the text box
            ax.text(
                0.95, 0.95, 
                stats_text,
                transform=ax.transAxes,
                va='top',
                ha='right',
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'),
                fontsize=8
            )
        
        # Add an overall title
        fig.suptitle(
            'Distribution of Cognitive and Emotional Measures Across Menopausal Stages', 
            fontsize=16, 
            y=0.98
        )
        
        # Adjust spacing
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(self.output_dir, 'outcome_violin_plots.png'), dpi=300)
        plt.close()
        
        print("Outcome violin plots saved.")

    def plot_stages_vs_scores(self):
        """
        Create a faceted plot showing scores across menopausal stages.
        Each subplot represents a different outcome measure, with stages on x-axis and scores on y-axis.
        Uses green color palette for lines and error bars.
        """
        # Prepare the data in long format
        plot_data = self.data.copy()
        
        # Ensure the categorical variable is properly ordered
        stage_order = ['Pre-menopause', 'Early Peri', 'Late Peri', 'Post-menopause', 'Surgical']
        plot_data['STATUS_Label'] = pd.Categorical(
            plot_data['STATUS_Label'],
            categories=stage_order,
            ordered=True
        )
        
        # Create a long-format dataframe for plotting
        melted_data = pd.melt(
            plot_data,
            id_vars=['SWANID', 'STATUS_Label'],
            value_vars=self.transformed_vars,  # Use transformed variables
            var_name='Measure',
            value_name='Score'
        )

        melted_data['Measure_Label'] = melted_data['Measure'].map(self.var_labels)
        
        # Calculate summary statistics for each stage and measure
        summary_data = melted_data.groupby(['Measure', 'Measure_Label', 'STATUS_Label']).agg(
            Mean=('Score', 'mean'),
            SE=('Score', lambda x: x.std() / np.sqrt(len(x))),
            Count=('Score', 'count')
        ).reset_index()
        
        # Create figure with subplots
        fig, axes = plt.subplots(4, 1, figsize=(14, 18), sharex=True)
        axes = axes.flatten()
        
        # Define green color for lines
        green_palette = sns.color_palette("YlGn", n_colors=8)
        selected_greens = [green_palette[3], green_palette[5]]
        line_color = selected_greens[0]  # Use a lighter green for lines
        errorbar_color = selected_greens[1]  # Use a darker green for error bars
        
        # Create plots for each measure
        for i, measure in enumerate(self.transformed_vars):
            ax = axes[i]
            
            # Get data for this measure
            measure_data = summary_data[summary_data['Measure'] == measure]
            
            # Set title even if no data
            ax.set_title(self.var_labels.get(measure, measure), fontsize=14)
            
            # Skip if no data for this measure
            if len(measure_data) == 0:
                ax.text(0.5, 0.5, f"No data available for {self.var_labels.get(measure, measure)}", 
                        ha='center', va='center', fontsize=12)
                continue
            
            # Plot mean and error bars
            ax.errorbar(
                x=measure_data['STATUS_Label'].cat.codes,  # Use category codes for x positioning
                y=measure_data['Mean'],
                yerr=measure_data['SE'] * 1.96,  # 95% CI
                fmt='o-',
                linewidth=2,
                markersize=8,
                capsize=5,
                color=line_color,
                ecolor=errorbar_color,
                label=self.var_labels.get(measure, measure)
            )
            
            # Add count labels for each status
            # Get a dataframe of counts for this measure
            measure_counts = melted_data[melted_data['Measure'] == measure].groupby('STATUS_Label').size().reset_index(name='Count')
            measure_counts = measure_counts.set_index('STATUS_Label')
            
            # Loop through every possible status to ensure we add counts for all
            for j, status in enumerate(stage_order):
                # Get the mean value for this status if it exists
                status_data = measure_data[measure_data['STATUS_Label'] == status]
                
                if not status_data.empty:
                    y_position = status_data['Mean'].values[0]
                    count = status_data['Count'].values[0]
                else:
                    # Use a default position if this status has no data
                    y_position = 0
                    # Try to get count from the full dataset
                    count = measure_counts.loc[status, 'Count'] if status in measure_counts.index else 0
                
                # Always add the count label
                ax.annotate(
                    f'n={count}',
                    xy=(j, y_position),
                    xytext=(0, 10),  # 10 points vertically offset
                    textcoords='offset points',
                    ha='center',
                    fontsize=18,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7)
                )
            
            # Set labels and title
            ax.set_title(self.var_labels.get(measure, measure), fontsize=18)
            
            # Set x-ticks and labels
            ax.set_xticks(range(len(stage_order)))
            ax.set_xticklabels(stage_order, rotation=45, ha='right', fontsize=18)
            ax.tick_params(axis='y', labelsize=18)
            
            # Add grid
            ax.grid(True, linestyle=':', alpha=0.6)
            
            # Add a horizontal line at y=0 if useful
            min_val = measure_data['Mean'].min() if not measure_data.empty else 0
            max_val = measure_data['Mean'].max() if not measure_data.empty else 0
            if min_val < 0 < max_val:
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Adjust spacing
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save the plot
        file_name = os.path.join(self.output_dir, 'stages_vs_scores_faceted.png')
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Faceted plot of stages vs. scores saved.")

    def create_all_visualizations(self):
        """Run all visualization methods."""
        
        print("Creating violin plots...")
        self.plot_violin_plots()

        print("Creating faceted plot of stages vs. scores...")
        self.plot_stages_vs_scores()
        
        print("All visualizations completed.")