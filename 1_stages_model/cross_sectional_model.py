import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pingouin as pg
import warnings
import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.save_output import OutputCapture, get_output_dir
from utils.data_validation import DataValidator

class MenopauseCognitionAnalysis:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path, low_memory=False)
        self.variables = ['TOTIDE1', 'TOTIDE2', 'NERVES', 'SAD', 'FEARFULA']
        self.output_dir = get_output_dir('1_stages_model', 'cross-sectional') 
        self.prepare_data()
    
    def prepare_data(self):
        """Include both natural and surgical menopause cases and create group labels."""
        self.data['STATUS'] = pd.to_numeric(self.data['STATUS'], errors='coerce')
        # Keep only statuses of interest (surgical: 1, 8; natural: 2,3,4,5)
        self.data = self.data[self.data['STATUS'].isin([1, 2, 3, 4, 5, 8])]
        
        # Map status to more descriptive labels for natural statuses and mark surgical statuses
        status_map = {
            1: 'Surgical',
            2: 'Post-menopause',
            3: 'Late Peri',
            4: 'Early Peri',
            5: 'Pre-menopause',
            8: 'Surgical'
        }
        self.data['STATUS_Label'] = self.data['STATUS'].map(status_map)
        
        # Create a new variable to distinguish natural vs. surgical menopause
        self.data['Menopause_Type'] = np.where(self.data['STATUS'].isin([1, 8]), 'Surgical', 'Natural')
        
        # If you still need an ordering for the natural stages, you can keep the categorical for STATUS_Label.
        # For instance, you may want natural stages to appear in order and have “Surgical” as a separate category.
        natural_order = ['Pre-menopause', 'Early Peri', 'Late Peri', 'Post-menopause']
        self.data['STATUS_Label'] = pd.Categorical(
            self.data['STATUS_Label'],
            categories=['Surgical'] + natural_order,
            ordered=True
        )

        # Validation
        print("\nPerforming data validation...")
        self.validation_results = self.validate_data()
        
        # Check validation results
        if  self.validation_results and 'missing_data' in  self.validation_results:
            # Check for high percentage of missing data
            for var, percentage in  self.validation_results['missing_data']['percentages'].items():
                if percentage > 20:
                    warnings.warn(f"{var} has {percentage:.1f}% missing data")
        
        # Check distributions if available
        if 'distributions' in self.validation_results:
            for var, results in self.validation_results['distributions'].items():
                if results and results.get('type') == 'numeric':
                    # Check skewness
                    if 'basic_stats' in results:
                        skewness = results['basic_stats'].get('skewness')
                        if skewness is not None and abs(skewness) > 2:
                            warnings.warn(f"{var} shows high skewness: {skewness:.2f}")
                    
                    # Check normality if test results exist
                    if ('normality_tests' in results and 
                        results['normality_tests'] and 
                        'dagostino_p' in results['normality_tests']):
                        p_value = results['normality_tests']['dagostino_p']
                        if p_value is not None and p_value < 0.05:
                            warnings.warn(f"{var} may not be normally distributed (p < 0.05)")
        
        # Check sample size warnings
        if  self.validation_results and 'sample_sizes' in  self.validation_results:
            if  self.validation_results['sample_sizes']['warnings']:
                for warning in  self.validation_results['sample_sizes']['warnings']:
                    warnings.warn(warning)

    def validate_data(self):
        """Perform comprehensive data validation."""
        # Only validate the specified variables
        validator = DataValidator(
            data=self.data,
            variables=self.variables,
            output_dir=self.output_dir
        )
        
        # Run validation checks
        validation_results = validator.run_checks(
            checks=['missing', 'distributions', 'group_sizes', 'homogeneity', 'multicollinearity', 'independence'],
            grouping_var='STATUS_Label'
        )
        return validation_results

    def perform_anova(self, variable):
        """
        Perform appropriate ANOVA test based on saved homogeneity results.
        Uses Welch's ANOVA with Games-Howell post-hoc when variances are not homogeneous.
        """
        try:
            # Check homogeneity from saved validation results
            homogeneity_results = self.validation_results.get('homogeneity', None)
            use_welch = (homogeneity_results and 
                        variable in homogeneity_results and 
                        homogeneity_results[variable] and 
                        not homogeneity_results[variable]['homogeneous'])
            
            # Prepare data for ANOVA
            clean_data = self.data[[variable, 'STATUS_Label']].dropna()
            
            if use_welch:
                # Perform Welch's ANOVA
                aov = pg.welch_anova(dv=variable, between='STATUS_Label', data=clean_data)
                # Perform Games-Howell post-hoc test
                posthoc = pg.pairwise_gameshowell(dv=variable, between='STATUS_Label', data=clean_data)
                test_type = "Welch's ANOVA"
            else:
                # Perform standard one-way ANOVA
                aov = pg.anova(dv=variable, between='STATUS_Label', data=clean_data)
                # Perform Tukey's HSD test
                posthoc = pg.pairwise_tukey(dv=variable, between='STATUS_Label', data=clean_data)
                test_type = "One-way ANOVA"
            
            # Get group descriptive statistics
            desc_stats = clean_data.groupby('STATUS_Label')[variable].agg(['count', 'mean', 'std']).round(3)
            
            # Create result dictionary
            result = {
                'test_type': test_type,
                'f_statistic': float(aov['F'].iloc[0]),
                'p_value': float(aov['p-unc'].iloc[0]),
                'posthoc_results': posthoc,
                'descriptive_stats': desc_stats
            }
            
            # Print detailed results
            print(f"\nPerforming {test_type} for {variable}:")
            print("\nANOVA results:")
            print(aov.round(4))
            print("\nDescriptive statistics:")
            print(desc_stats)
            print("\nPost-hoc test results:")
            print(posthoc.round(4))
            
            return result
            
        except Exception as e:
            print(f"Error in ANOVA calculation: {e}")
            return None

    def calculate_summary_stats(self, variable):
        """Calculate summary statistics for raw scores at each stage."""
        self.data[variable] = pd.to_numeric(self.data[variable], errors='coerce')  # Forces conversion, NaNs for invalid values
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
                print(f"\{anova_results['test_type']} results:")
                print(f"F-statistic: {anova_results['f_statistic']:.3f}")
                print(f"p-value: {anova_results['p_value']:.3e}")
                print("\nPost Hoc results:")
                print(anova_results['posthoc_results'])
            
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
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up output capture
        output_capture = OutputCapture(self.output_dir, 'analysis_results.txt')
        sys.stdout = output_capture

        try:
            print("Creating violin plots with box plots and performing ANOVA...")
            self.plot_violin_with_box(self.variables)
            
            print("\nCreating mean trend plots...")
            self.plot_mean_trends(self.variables)

            print("\nAnalysis complete.")

        finally:
            sys.stdout = output_capture.terminal
            output_capture.close()

if __name__ == "__main__":
    # Initialize analysis with your data file
    analysis = MenopauseCognitionAnalysis("processed_combined_data.csv")
    
    # Run the complete analysis
    analysis.run_complete_analysis()