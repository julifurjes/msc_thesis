import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.save_output import OutputCapture, get_output_dir 

class MenopauseScoreAnalysis:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path, low_memory=False)
        self.prepare_data()
        self.output_dir = get_output_dir('1_stages_model', 'overall') 
    
    def prepare_data(self):
        """Prepare the dataset with proper status labels and variable conversion."""
        # Convert STATUS to numeric
        self.data['STATUS'] = pd.to_numeric(self.data['STATUS'], errors='coerce')
        self.data = self.data[self.data['STATUS'].between(2, 5)]
        
        # Map STATUS to descriptive labels
        status_map = {
            2: 'Post-menopause',
            3: 'Late Peri',
            4: 'Early Peri',
            5: 'Pre-menopause'
        }
        self.data['STATUS_Label'] = self.data['STATUS'].map(status_map)
        
        # Create categorical order for plotting
        status_order = ['Pre-menopause', 'Early Peri', 'Late Peri', 'Post-menopause']
        self.data['STATUS_Label'] = pd.Categorical(
            self.data['STATUS_Label'],
            categories=status_order,
            ordered=True
        )
        
        # Convert score variables to numeric and handle missing values
        score_vars = ['TOTIDE1', 'TOTIDE2', 'NERVES', 'SAD', 'FEARFULA']
        for var in score_vars:
            self.data[var] = pd.to_numeric(self.data[var], errors='coerce')
            # Remove NaN values for each variable
            self.data[var] = self.data[var].fillna(self.data[var].mean())

    def perform_anova(self, variable):
        """Perform one-way ANOVA and Tukey's HSD test for a given variable."""
        # Create groups for ANOVA, dropping NaN values
        groups = []
        labels = []
        for name, group in self.data.groupby('STATUS_Label', observed=True)[variable]:
            # Remove any NaN values
            clean_group = group.dropna()
            if len(clean_group) > 0:
                groups.append(clean_group.values)
                labels.append(name)
        
        # Perform one-way ANOVA
        try:
            f_stat, p_value = f_oneway(*groups)
            
            # Prepare data for Tukey's test
            all_data = np.concatenate(groups)
            group_labels = np.concatenate([[label] * len(group) for label, group in zip(labels, groups)])
            
            # Perform Tukey's HSD test
            tukey = pairwise_tukeyhsd(all_data, group_labels)
            
            return {
                'f_statistic': f_stat,
                'p_value': p_value,
                'tukey_results': tukey
            }
        except Exception as e:
            print(f"Error in ANOVA calculation: {e}")
            return None

    def calculate_summary_stats(self, variable):
        """Calculate summary statistics for raw scores at each stage."""
        summary = self.data.groupby('STATUS_Label', observed=True)[variable].agg([
            'count',
            'mean',
            'std',
            lambda x: stats.sem(x.dropna()),  # Standard error of mean
            'median',
            lambda x: np.percentile(x.dropna(), 25),  # Q1
            lambda x: np.percentile(x.dropna(), 75),  # Q3
        ]).round(3)
        summary.columns = ['n', 'mean', 'std', 'sem', 'median', 'Q1', 'Q3']
        return summary

    def plot_violin_with_box(self, variables):
        """Create violin plots with box plot overlays for raw scores."""
        var_labels = {
            'TOTIDE1': 'Total IDE Score 1',
            'TOTIDE2': 'Total IDE Score 2',
            'NERVES': 'Nervousness Score',
            'SAD': 'Sadness Score',
            'FEARFULA': 'Fearfulness Score'
        }

        for var in variables:
            # Print detailed statistics and ANOVA results
            print(f"\nAnalysis for {var_labels[var]}:")
            stats = self.calculate_summary_stats(var)
            print("\nSummary Statistics:")
            print(stats)
            
            # Perform and print ANOVA results
            anova_results = self.perform_anova(var)
            if anova_results:
                print(f"\nOne-way ANOVA results:")
                print(f"F-statistic: {anova_results['f_statistic']:.3f}")
                print(f"p-value: {anova_results['p_value']:.3e}")
                print("\nTukey's HSD test results:")
                print(anova_results['tukey_results'])
            
            # Create the plot
            plt.figure(figsize=(12, 6))
            
            # Create violin plot with box plot overlay
            sns.violinplot(data=self.data, x='STATUS_Label', y=var,
                         hue='STATUS_Label', inner='box', palette='coolwarm',
                         legend=False)
            
            # Add summary statistics and ANOVA results
            stat_text = "Summary Statistics & ANOVA:\n"
            for idx, row in stats.iterrows():
                stat_text += f"\n{idx}:\n"
                stat_text += f"n={int(row['n'])}, mean={row['mean']:.2f}\n"
                stat_text += f"median={row['median']:.2f}, sd={row['std']:.2f}\n"
            if anova_results:
                stat_text += f"\nANOVA p-value: {anova_results['p_value']:.3e}"
            
            plt.text(1.15, 0.5, stat_text,
                    transform=plt.gca().transAxes,
                    bbox=dict(facecolor='white', alpha=0.8),
                    fontsize=8,
                    verticalalignment='center')
            
            title = f'Distribution of {var_labels[var]} Across Menopausal Stages'
            if anova_results:
                title += f'\n(ANOVA: F={anova_results["f_statistic"]:.2f}, p={anova_results["p_value"]:.3e})'
            plt.title(title)
            plt.xlabel('Menopausal Status (Timeline →)')
            plt.ylabel(f'{var_labels[var]}')
            plt.xticks(rotation=45)
            
            plt.tight_layout()

            file_name = f'{self.output_dir}/{var}_violin_plot.png'
            plt.savefig(file_name)

            plt.close()

    def plot_mean_trends(self, variables):
        """Create mean plots with error bars for raw scores."""
        var_labels = {
            'TOTIDE1': 'Total IDE Score 1',
            'TOTIDE2': 'Total IDE Score 2',
            'NERVES': 'Nervousness Score',
            'SAD': 'Sadness Score',
            'FEARFULA': 'Fearfulness Score'
        }
        
        for var in variables:
            # Calculate and print summary statistics
            stats = self.calculate_summary_stats(var)
            anova_results = self.perform_anova(var)
            
            print(f"\nMean trend statistics for {var_labels[var]}:")
            print("\nStage-by-stage values:")
            for idx, row in stats.iterrows():
                print(f"{idx}:")
                print(f"  Mean: {row['mean']:.3f}")
                print(f"  SEM: ±{row['sem']:.3f}")
                print(f"  N: {int(row['n'])}")
            
            # Create the plot
            plt.figure(figsize=(10, 6))
            
            # Plot means and error bars
            x = range(len(stats.index))
            plt.errorbar(x, stats['mean'], yerr=stats['sem'],
                        fmt='o-', capsize=5, capthick=2,
                        linewidth=2, markersize=8,
                        color='blue', ecolor='black')
            
            title = f'Mean {var_labels[var]} Across Menopausal Stages'
            if anova_results:
                title += f'\n(ANOVA: F={anova_results["f_statistic"]:.2f}, p={anova_results["p_value"]:.3e})'
            plt.title(title)
            plt.xlabel('Menopausal Status (Timeline →)')
            plt.ylabel(f'Mean {var_labels[var]} (± SEM)')
            plt.xticks(x, stats.index, rotation=45)
            
            # Add grid
            plt.grid(True, linestyle='--', alpha=0.3)
            
            # Add value labels
            for i, (mean, sem) in enumerate(zip(stats['mean'], stats['sem'])):
                plt.text(i, mean + sem + 0.1, f'{mean:.2f}',
                        ha='center', va='bottom')
            
            plt.tight_layout()

            file_name = f'{self.output_dir}/{var}_mean_trend_plot.png'
            plt.savefig(file_name)

            plt.close()

    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        variables = ['TOTIDE1', 'TOTIDE2', 'NERVES', 'SAD', 'FEARFULA']

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up output capture
        output_capture = OutputCapture(self.output_dir)
        sys.stdout = output_capture

        try:
            print("Creating violin plots with box plots and performing ANOVA...")
            self.plot_violin_with_box(variables)
            
            print("\nCreating mean trend plots...")
            self.plot_mean_trends(variables)

            print("\nAnalysis complete.")

        finally:
            sys.stdout = output_capture.terminal
            output_capture.close()

if __name__ == "__main__":
    # Initialize analysis with your data file
    analysis = MenopauseScoreAnalysis("processed_combined_data.csv")
    
    # Run the complete analysis
    analysis.run_complete_analysis()