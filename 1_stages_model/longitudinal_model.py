import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import gee
from statsmodels.genmod.cov_struct import Exchangeable
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
        self.cognitive_vars = ['TOTIDE1', 'TOTIDE2', 'NERVES', 'SAD', 'FEARFULA']
        self.filter_status()
        self.gee_results = {}
        self.output_dir = get_output_dir('1_stages_model', 'longitudinal')
    
    def filter_status(self):
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

    def run_data_validation(self):
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
        variables_to_check = self.cognitive_vars + ['STATUS_Label']
        # Only validate the specified variables
        validator = DataValidator(
            data=self.data,
            variables=variables_to_check,
            output_dir=self.output_dir
        )
        
        # Run validation checks
        validation_results = validator.run_checks(
            checks=['missing', 'distributions', 'group_sizes', 'multicollinearity', 'independence'],
            grouping_var='STATUS_Label'
        )
        return validation_results

    def run_gee_analysis(self, covariates=None):
        """
        Run GEE analysis for each cognitive variable.
        
        Parameters:
        -----------
        covariates : list
            List of additional covariate column names to include in the model
        """
        cognitive_vars = ['TOTIDE1', 'TOTIDE2', 'NERVES', 'SAD', 'FEARFULA']
        
        # Ensure numeric types for cognitive variables and covariates
        for var in cognitive_vars:
            self.data[var] = pd.to_numeric(self.data[var], errors='coerce')
        
        if covariates:
            for var in covariates:
                self.data[var] = pd.to_numeric(self.data[var], errors='coerce')
        
        # Prepare covariates string for formula
        if covariates is None:
            covariates = []
        
        # Handle covariates differently - don't use categorical encoding for continuous variables
        covariate_str = ' + '.join(covariates) if covariates else ''
        
        # Dictionary to store results
        self.gee_results = {}
        
        for outcome in cognitive_vars:
            # Create formula using STATUS as categorical variable
            if covariate_str:
                formula = f"{outcome} ~ C(STATUS_Label, Treatment('Pre-menopause')) + {covariate_str}"
            else:
                formula = f"{outcome} ~ C(STATUS_Label, Treatment('Pre-menopause'))"
            
            try:
                # Drop rows with missing values
                analysis_data = self.data.dropna(subset=[outcome] + covariates if covariates else [outcome])
                
                # Fit GEE model
                model = gee(
                    formula=formula,
                    groups="SWANID",
                    data=analysis_data,
                    cov_struct=Exchangeable(),
                    family=sm.families.Gaussian()
                )
                
                results = model.fit()
                self.gee_results[outcome] = results
                
                # Print detailed results
                print(f"\nGEE Results for {outcome}")
                print("=" * 50)
                print(results.summary())
                
                # Calculate and store effect sizes (standardized coefficients)
                y_std = analysis_data[outcome].std()
                
                # Get the standard deviations for the predictors
                predictor_stds = {}
                for pred in results.model.exog_names:
                    if pred == 'Intercept':
                        predictor_stds[pred] = 1
                    elif pred.startswith('C(STATUS_Label'):
                        predictor_stds[pred] = 1  # Binary variables
                    else:
                        predictor_stds[pred] = analysis_data[pred].std()
                
                # Calculate standardized coefficients
                standardized_params = pd.Series({
                    name: param * (predictor_stds.get(name, 1) / y_std)
                    for name, param in results.params.items()
                })
                
                print("\nStandardized Coefficients:")
                print(standardized_params)
                
            except Exception as e:
                print(f"Error in GEE analysis for {outcome}: {str(e)}")
                print("Data shape:", analysis_data.shape)
                print("Formula:", formula)
    
    def plot_gee_results(self):
        """Create forest plots of GEE results for all cognitive variables."""
        if not self.gee_results:
            print("No GEE results available. Run run_gee_analysis() first.")
            return
        
        # Create figure
        fig, axes = plt.subplots(len(self.gee_results), 1, figsize=(12, 4*len(self.gee_results)))
        if len(self.gee_results) == 1:
            axes = [axes]
        
        # Define the status effects we want to plot
        status_effects = ['Early Peri', 'Late Peri', 'Post-menopause']
        
        for idx, (outcome, results) in enumerate(self.gee_results.items()):
            print(f"\nProcessing {outcome}...")
            
            # Initialize lists to store plot data
            coefs = []
            errors = []
            pvalues = []
            names = []
            
            # Extract coefficients using the exact parameter names from the GEE output
            for status in status_effects:
                param_name = f"C(STATUS_Label, Treatment('Pre-menopause'))[T.{status}]"
                
                if param_name in results.params.index:
                    print(f"Found coefficient for {status}: {results.params[param_name]:.3f}")
                    coefs.append(results.params[param_name])
                    errors.append(results.bse[param_name])
                    pvalues.append(results.pvalues[param_name])
                    names.append(status)
                else:
                    print(f"Warning: No coefficient found for {status} using {param_name}")
            
            if not coefs:
                print(f"No coefficients found for {outcome}")
                continue
            
            # Create forest plot
            y_pos = np.arange(len(names))
            axes[idx].errorbar(
                coefs, y_pos,
                xerr=1.96 * np.array(errors),  # 95% CI
                fmt='o',
                capsize=5,
                markersize=8,
                elinewidth=2,
                capthick=2
            )
            
            # Add vertical line at zero
            axes[idx].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            
            # Customize plot
            axes[idx].set_yticks(y_pos)
            axes[idx].set_yticklabels(names)
            axes[idx].set_title(f'{outcome} Effects\n(Reference: Pre-menopause)', pad=20)
            axes[idx].set_xlabel('Coefficient Value')
            
            # Add coefficient values and p-values as text
            for i, (coef, p) in enumerate(zip(coefs, pvalues)):
                # Add asterisks for significance levels
                stars = ''
                if p < 0.001:
                    stars = '***'
                elif p < 0.01:
                    stars = '**'
                elif p < 0.05:
                    stars = '*'
                
                text = f' {coef:.3f}{stars}\n(p={p:.3f})'
                axes[idx].text(
                    coef, i,
                    text,
                    verticalalignment='center',
                    horizontalalignment='left' if coef >= 0 else 'right',
                    fontsize=10
                )
            
            # Add gridlines
            axes[idx].grid(True, axis='x', linestyle=':', alpha=0.6)
            
            # Adjust x-axis to make sure all error bars and text are visible
            x_max = max([abs(c + e*1.96) for c, e in zip(coefs, errors)])
            axes[idx].set_xlim(-x_max*1.2, x_max*1.2)
        
        # Add legend for significance levels
        fig.text(0.99, 0.01, 
                '* p<0.05   ** p<0.01   *** p<0.001',
                fontsize=10,
                horizontalalignment='right')
        
        plt.tight_layout()
        
        # Save the plot
        file_name = os.path.join(self.output_dir, 'gee_results_forest_plots.png')
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nPlot saved as: {file_name}")

    def calculate_baseline_changes(self):
        """Calculate changes from each subject's first available measurement as baseline."""
        for col in self.cognitive_vars:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        # Sort by subject ID and visit
        self.data = self.data.sort_values(['SWANID', 'VISIT'])
        
        # Use the first visit for each subject as baseline, regardless of STATUS
        baseline_data = self.data.groupby('SWANID').first().reset_index()
        
        baseline_cols = {col: f'{col}_baseline' for col in self.cognitive_vars}
        baseline_data = baseline_data[['SWANID'] + self.cognitive_vars].rename(columns=baseline_cols)
        
        # Merge baseline data back into main DataFrame
        self.data = self.data.merge(baseline_data, on='SWANID', how='left')
        
        # Calculate changes from baseline
        for var in self.cognitive_vars:
            self.data[f'{var}_change'] = self.data[var] - self.data[f'{var}_baseline']
            self.data[f'{var}_pct_change'] = (
                (self.data[var] - self.data[f'{var}_baseline']) / self.data[f'{var}_baseline'] * 100
            )
        
        return self.data

    def plot_change_distributions(self, use_percentage=False):
        """Create violin plots with embedded box plots for change scores."""
        var_labels = {
            'TOTIDE1': 'Total IDE Score 1',
            'TOTIDE2': 'Total IDE Score 2',
            'NERVES': 'Nervousness Score',
            'SAD': 'Sadness Score',
            'FEARFULA': 'Fearfulness Score'
        }
        
        # Create a figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 20))
        axes = axes.flatten()  # Flatten the 2D array of axes
        
        # Set color palette
        colors = sns.color_palette("coolwarm", n_colors=4)
        
        for idx, var in enumerate(self.cognitive_vars):
            suffix = '_pct_change' if use_percentage else '_change'
            var_change = var + suffix
            
            # Create violin plot with boxplot inside
            sns.violinplot(
                data=self.data,
                x='STATUS_Label',  # or another variable if you want to separate by VISIT/time
                y=var_change,
                hue='Menopause_Type',
                split=True,  # if appropriate
                inner='box',
                ax=axes[idx],
                palette="coolwarm"
            )
            
            # Customize the plot
            axes[idx].set_title(f'Changes in {var_labels[var]} Across Menopausal Stages', pad=20)
            axes[idx].set_xlabel('Menopausal Status (Timeline →)', labelpad=10)
            axes[idx].set_ylabel(
                'Percent Change from Baseline' if use_percentage 
                else 'Absolute Change from Baseline'
            )
            
            # Add horizontal line at y=0 for reference
            axes[idx].axhline(y=0, color='black', linestyle='--', alpha=0.3)
            
            # Rotate x-axis labels for better readability
            axes[idx].tick_params(axis='x', rotation=45)
            
            # Add summary statistics as text
            summary_stats = self.data.groupby('STATUS_Label', observed=True)[var_change].agg([
                'count', 'mean', 'std', 'median'
            ]).round(2)
            
            stat_text = "Summary Statistics:\n"
            for status, stats in summary_stats.iterrows():
                stat_text += f"\n{status}:\n"
                stat_text += f"n={stats['count']}, mean={stats['mean']}\n"
                stat_text += f"median={stats['median']}, sd={stats['std']}\n"
            
            # Position the text box in a better location
            text_box = axes[idx].text(
                1.15,  # x position
                0.5,   # y position
                stat_text,
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.8),
                transform=axes[idx].transAxes,  # Use axes coordinates
                verticalalignment='center'
            )
        
        # Remove the empty subplot if we have one
        if len(self.cognitive_vars) < len(axes):
            fig.delaxes(axes[-1])
        
        # Add a main title
        fig.suptitle(
            'Cognitive Changes Across Menopausal Stages\n'
            '(Pre-menopause → Early Peri → Late Peri → Post-menopause)',
            y=1.02,
            fontsize=14
        )
        
        # Adjust layout and display
        plt.tight_layout()

        file_name = os.path.join(self.output_dir, f'menopause_changes_{"percent" if use_percentage else "absolute"}.png')
        # Save the plot
        fig.savefig(
            file_name,
            dpi=300,
            bbox_inches='tight'
        )
        
        plt.close()

    def run_complete_analysis(self, covariates=None):
        """Run the complete analysis pipeline including GEE analysis."""
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up output capture
        output_capture = OutputCapture(self.output_dir)
        sys.stdout = output_capture

        try:
            print("Running data validation...")
            self.run_data_validation()

            print("Calculating baseline changes...")
            self.calculate_baseline_changes()
            
            print("\nRunning GEE analysis...")
            self.run_gee_analysis(covariates=covariates)
            
            print("\nPlotting GEE results...")
            self.plot_gee_results()
            
            print("\nCreating distribution plots for absolute changes...")
            self.plot_change_distributions(use_percentage=False)
            
            print("\nCreating distribution plots for percentage changes...")
            self.plot_change_distributions(use_percentage=True)

            print("\nAnalysis complete.")

        finally:
            sys.stdout = output_capture.terminal
            output_capture.close()

if __name__ == "__main__":
    # Initialize analysis with your data file
    analysis = MenopauseCognitionAnalysis("processed_combined_data.csv")
    
    # Run the complete analysis with optional covariates
    covariates = ['AGE']
    analysis.run_complete_analysis(covariates=covariates)