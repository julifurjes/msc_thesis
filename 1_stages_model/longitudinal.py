import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
import os
import sys

from visualisations import MenopauseVisualisations
from proportion_analysis import MenopauseDeclineAnalysis

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.save_output import OutputCapture, get_output_dir

class MenopauseCognitionAnalysis:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path, low_memory=False)
        self.outcome_vars = ['TOTIDE1', 'TOTIDE2', 'NERVES', 'SAD', 'FEARFULA']
        self.mixed_model_results = {}
        self.output_dir = get_output_dir('1_stages_model', 'longitudinal')

    def transform_variables(self):
        """Apply transformations to address heteroscedasticity and multicollinearity."""
        # Log transformation for heavily skewed variables
        self.data['NERVES_log'] = np.log1p(self.data['NERVES'])  # log(x+1)

        # Average for TOTIDE to reduce multicollinearity
        self.data['TOTIDE_avg'] = (self.data['TOTIDE1'] + self.data['TOTIDE2']) / 2
        
        # Square root for moderately skewed variables
        self.data['SAD_sqrt'] = np.sqrt(self.data['SAD'])
        self.data['FEARFULA_sqrt'] = np.sqrt(self.data['FEARFULA'])

        self.outcome_vars = ['TOTIDE_avg', 'NERVES_log', 'SAD_sqrt', 'FEARFULA_sqrt']

        # Make langauge categorical
        self.data['LANGCOG'] = self.data['LANGCOG'].astype('category')
    
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

        natural_order = ['Pre-menopause', 'Early Peri', 'Late Peri', 'Post-menopause']
        self.data['STATUS_Label'] = pd.Categorical(
            self.data['STATUS_Label'],
            categories=['Surgical'] + natural_order,
            ordered=True
        )

    def run_mixed_models(self):
        """
        Run linear mixed-effects models for each cognitive variable.
        This model includes random intercepts for each subject.
        """

        # Get the reference language for the model (the most common one)
        reference_language = self.data['LANGCOG'].mode()[0]
        
        # Ensure numeric types for cognitive variables
        for var in self.outcome_vars:
            self.data[var] = pd.to_numeric(self.data[var], errors='coerce')
        
        covariates = ['AGE', 'LANGCOG']
        
        # Dictionary to store results
        self.mixed_model_results = {}
        
        for outcome in self.outcome_vars:
            # Create formula
            covariate_formula = f"AGE + C(LANGCOG, Treatment({reference_language}))"
            formula = (f"{outcome} ~ C(STATUS_Label, Treatment('Pre-menopause')) + VISIT + {covariate_formula}")
            
            try:
                # Drop rows with missing values
                analysis_data = self.data.dropna(subset=[outcome] + covariates if covariates else [outcome])

                # Handling unbalanced data with weights
                status_counts = analysis_data['STATUS_Label'].value_counts()
                analysis_data['weights'] = analysis_data['STATUS_Label'].map(
                    lambda x: 1 / (status_counts[x] / sum(status_counts))
                )
                
                # Fit mixed model with random intercept for SWANID
                model = mixedlm(
                    formula=formula,
                    groups=analysis_data["SWANID"],
                    data=analysis_data,
                    re_formula="~VISIT"
                )
                
                # Add weights and fit the model
                results = model.fit(reml=True, weights=analysis_data['weights'])
                self.mixed_model_results[outcome] = results
                
                # Print detailed results
                print(f"\nMixed Model Results for {outcome}")
                print("=" * 50)
                print(results.summary())
                
                # Calculate marginal and conditional R-squared
                try:
                    # Residual variance
                    resid_var = results.scale
                    # Random effects variance
                    re_var = results.cov_re.iloc[0, 0] if hasattr(results.cov_re, 'iloc') else results.cov_re[0][0]
                    # Variance explained by fixed effects (approximate)
                    var_fixed = np.var(results.predict(analysis_data))
                    
                    # Marginal R² (fixed effects only)
                    marginal_r2 = var_fixed / (var_fixed + re_var + resid_var)
                    # Conditional R² (both fixed and random effects)
                    conditional_r2 = (var_fixed + re_var) / (var_fixed + re_var + resid_var)
                    
                    print(f"\nApproximate Marginal R² (fixed effects): {marginal_r2:.4f}")
                    print(f"Approximate Conditional R² (fixed + random): {conditional_r2:.4f}")
                    
                except Exception as e:
                    print(f"Error calculating R-squared: {str(e)}")
                
                # Check residuals and model diagnostics
                self.check_model_diagnostics(results, outcome, analysis_data)
                
            except Exception as e:
                print(f"Error in mixed model analysis for {outcome}: {str(e)}")
    
    def check_model_diagnostics(self, model_results, outcome, data):
        """Check model residuals and diagnostics for the mixed model."""
        try:
            # Calculate residuals
            predicted = model_results.predict(data)
            actual = data[outcome]
            residuals = actual - predicted
            
            # Plot residuals
            plt.figure(figsize=(12, 10))
            
            plt.subplot(2, 2, 1)
            plt.scatter(predicted, residuals, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='-')
            plt.title(f'Residuals vs Fitted for {outcome}')
            plt.xlabel('Fitted values')
            plt.ylabel('Residuals')
            
            plt.subplot(2, 2, 2)
            sns.histplot(residuals, kde=True)
            plt.title('Histogram of Residuals')
            plt.xlabel('Residual Value')
            
            plt.subplot(2, 2, 3)
            sm.qqplot(residuals.dropna(), line='s', ax=plt.gca())
            plt.title('Q-Q Plot of Residuals')
            
            plt.subplot(2, 2, 4)
            plt.scatter(range(len(residuals)), residuals, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='-')
            plt.title('Residuals vs Order')
            plt.xlabel('Observation Order')
            plt.ylabel('Residuals')
            
            # Save the plot
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{outcome}_mixed_diagnostics.png'))
            plt.close()
            
        except Exception as e:
            print(f"Error in model diagnostics: {str(e)}")

    def _calculate_reasonable_limits(self, coefs, errors, percentile=95):
        """
        Calculate reasonable axis limits that won't be distorted by extreme outliers.
        Uses a percentile-based approach to avoid extremely wide confidence intervals.
        """
        # Calculate ends of all error bars
        all_ends = []
        for coef, error in zip(coefs, errors):
            all_ends.append(coef + (error * 1.96))
            all_ends.append(coef - (error * 1.96))
        
        # Determine reasonable limits based on percentiles
        if all_ends:
            # Get the percentile values (exclude extreme outliers)
            min_val = np.percentile(all_ends, 100 - percentile)
            max_val = np.percentile(all_ends, percentile)
            
            # Ensure zero is included in the range
            min_val = min(min_val, 0)
            max_val = max(max_val, 0)
            
            # Add padding
            range_val = max_val - min_val
            min_val -= range_val * 0.1
            max_val += range_val * 0.1
            
            return min_val, max_val
        
        return -1, 1  # Default if no data

    def plot_forest_plot_from_models(self):
        """
        Create a forest plot using the actual model results from the mixed-effects models.
        Features bounded error bars and optimized label placement.
        """
        if not self.mixed_model_results:
            print("No mixed model results available. Run run_mixed_models() first.")
            return
        
        # Set up nice labels for the measures
        measure_labels = {
            'TOTIDE_avg': 'Cognitive Performance',
            'NERVES_log': 'Nervousness',
            'SAD_sqrt': 'Sadness', 
            'FEARFULA_sqrt': 'Fearfulness'
        }
        
        # Define the status effects we want to plot
        status_effects = ['Early Peri', 'Late Peri', 'Post-menopause', 'Surgical']
        
        # Create the figure
        fig, axes = plt.subplots(4, 1, figsize=(14, 20), sharex=False)  # sharex=False to allow different x scales
        axes = axes.flatten()
        
        # Import the palette and select distinct colors
        green_palette = sns.color_palette("YlGn", n_colors=8)  # Create a larger palette to select from
        selected_greens = [green_palette[1], green_palette[3], green_palette[5], green_palette[7]]  # Take every second color

        # Define color mapping for p-values using selected colors from YlGn palette
        def get_color(p_value):
            if p_value < 0.001:
                return selected_greens[3], '***'  # Most significant: darkest green
            elif p_value < 0.01:
                return selected_greens[2], '**'   # Very significant: medium-dark green
            elif p_value < 0.05:
                return selected_greens[1], '*'    # Significant: light-medium green
            else:
                return 'gray', ''              # Not significant: gray
        
        # Plot each outcome measure
        for i, (outcome, results) in enumerate(self.mixed_model_results.items()):
            ax = axes[i]
            
            # Initialize lists to store plot data
            coefs = []
            errors = []
            pvalues = [] 
            names = []
            
            # Extract coefficients from the model results
            for status in status_effects:
                param_name = f"C(STATUS_Label, Treatment('Pre-menopause'))[T.{status}]"
                
                if param_name in results.params.index:
                    print(f"Found coefficient for {status}: {results.params[param_name]:.3f}")
                    coefs.append(results.params[param_name])
                    errors.append(results.bse[param_name])
                    pvalues.append(results.pvalues[param_name])
                    names.append(status)
                else:
                    print(f"Warning: No coefficient found for {status}")
            
            if not coefs:
                print(f"No coefficients found for {outcome}")
                ax.set_visible(False)
                continue
            
            # Plot horizontal error bars
            y_positions = np.arange(len(names))
            
            # Calculate reasonable x-axis limits based on percentiles
            min_bound, max_bound = self._calculate_reasonable_limits(coefs, errors, percentile=95)
            ax.set_xlim(min_bound, max_bound)
            
            # Plot each point with color based on significance
            for j, (y, coef, error, p) in enumerate(zip(y_positions, coefs, errors, pvalues)):
                color, marker = get_color(p)
                
                # Calculate error bar ends but constrain them to the plot limits
                lower_error = max(min_bound * 0.95, coef - error * 1.96)
                upper_error = min(max_bound * 0.95, coef + error * 1.96)
                
                # Adjust xerr to create asymmetric error bars if needed
                left_err = coef - lower_error
                right_err = upper_error - coef
                
                # Plot with potentially asymmetric error bars
                ax.errorbar(
                    x=coef, 
                    y=y,
                    xerr=[[left_err], [right_err]],  # Asymmetric error bars
                    fmt='o',
                    color=color,
                    capsize=5,
                    markersize=8,
                    elinewidth=2,
                    capthick=2
                )
                
                # Calculate where to place the label
                label_x = upper_error + (max_bound - min_bound) * 0.02 + 0.01
                
                # Add coefficient value and significance markers
                ax.text(
                    label_x,
                    y,
                    f'{coef:.3f} {marker}',
                    va='center',
                    ha='right',
                    color=color,
                    fontweight='bold' if p < 0.05 else 'normal',
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2', edgecolor='none')
                )
                
                # If error bars were trimmed, indicate this with an arrow
                if coef - error * 1.96 < min_bound * 0.95:
                    ax.annotate('', xy=(min_bound * 0.98, y), xytext=(min_bound * 0.9, y),
                            arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
                
                if coef + error * 1.96 > max_bound * 0.95:
                    ax.annotate('', xy=(max_bound * 0.98, y), xytext=(max_bound * 0.9, y),
                            arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
            
            # Add reference line at zero
            ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            
            # Set y-ticks and labels
            ax.set_yticks(y_positions)
            ax.set_yticklabels(names, fontsize=18)
            
            # Set title and labels
            ax.set_title(measure_labels.get(outcome, outcome), fontsize=18)
            
            # Make grid lines more subtle
            ax.grid(True, linestyle=':', alpha=0.4)
        
        # Add a legend for significance
        fig.text(
            0.5, 0.01, 
            '* p<0.05   ** p<0.01   *** p<0.001', 
            ha='center', 
            fontsize=12
        )
        
        # Adjust spacing - increase wspace for wider spacing between subplots
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.subplots_adjust(wspace=0.3)
        
        # Save the plot with higher DPI for better quality
        file_name = os.path.join(self.output_dir, 'model_forest_plot.png')
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Forest plot saved.")

    def run_complete_analysis(self):
        """Run the complete analysis pipeline with mixed models."""
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        print("Transforming variables for analysis...")
        self.transform_variables()

        print("Filtering menopausal status...")
        self.filter_status()
        
        # Set up output capture
        output_capture = OutputCapture(self.output_dir)
        sys.stdout = output_capture

        try:
            print("\nRunning linear mixed-effects models...")
            self.run_mixed_models()
            self.plot_forest_plot_from_models()

            print("\nAnalysis complete.")

        finally:
            sys.stdout = output_capture.terminal
            output_capture.close()

if __name__ == "__main__":
    # Initialize analysis with your data file
    analysis = MenopauseCognitionAnalysis("processed_combined_data.csv")
    analysis.run_complete_analysis()
    proportion_analysis = MenopauseDeclineAnalysis(analysis.data)
    proportion_analysis.run_analysis()
    viz = MenopauseVisualisations(analysis.data)
    viz.create_all_visualizations()