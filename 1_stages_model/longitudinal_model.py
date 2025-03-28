import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
import warnings
import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.save_output import OutputCapture, get_output_dir
from utils.further_checks import FurtherChecks

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
        # For instance, you may want natural stages to appear in order and have "Surgical" as a separate category.
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
        
        # Ensure numeric types for cognitive variables
        for var in self.outcome_vars:
            self.data[var] = pd.to_numeric(self.data[var], errors='coerce')
        
        covariates = ['AGE']
        
        # Handle covariates 
        covariate_str = ' + '.join(covariates) if covariates else ''
        
        # Dictionary to store results
        self.mixed_model_results = {}
        
        for outcome in self.outcome_vars:
            # Create formula
            formula = (f"{outcome} ~ C(STATUS_Label, Treatment('Pre-menopause')) + VISIT + {covariate_str}")
            
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
                    # Calculate total variance
                    total_var = resid_var + re_var
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
                print("Data shape:", analysis_data.shape)
                print("Formula:", formula)
    
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
    
    def plot_forest_plots(self):
        """Create forest plots of mixed model results for all cognitive variables."""
        if not self.mixed_model_results:
            print("No mixed model results available. Run run_mixed_models() first.")
            return
        
        # Calculate number of plots and arrange in 2 columns
        n_plots = len(self.mixed_model_results)
        n_cols = 2  # Always use 2 columns
        n_rows = (n_plots + 1) // 2  # Ceiling division for odd number of plots
        
        # Create a figure with subplots in a 2-column grid
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
        axes = axes.flatten()  # Flatten to make indexing easier
        
        # Define the status effects we want to plot
        status_effects = ['Early Peri', 'Late Peri', 'Post-menopause']
        
        plot_idx = 0
        for outcome, results in self.mixed_model_results.items():
            print(f"\nProcessing {outcome}...")
            ax = axes[plot_idx]
            plot_idx += 1
            
            # Initialize lists to store plot data
            coefs = []
            errors = []
            pvalues = []
            names = []
            
            # Extract coefficients
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
            
            # Create forest plot with spaced points
            y_pos = np.arange(len(names)) * 1.5
            
            ax.errorbar(
                coefs, y_pos,
                xerr=1.96 * np.array(errors),  # 95% CI
                fmt='o',
                capsize=5,
                markersize=8,
                elinewidth=2,
                capthick=2
            )
            
            # Add vertical line at zero
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            
            # Customize plot
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, fontsize=12)
            ax.set_title(f'{outcome} Effects (Mixed Model)\n(Reference: Pre-menopause)', fontsize=14)
            ax.set_xlabel('Coefficient Value', fontsize=12)
            
            # Calculate x-axis limits first to ensure text stays within bounds
            x_max = max([abs(c + e*1.96*1.2) for c, e in zip(coefs, errors)])
            ax.set_xlim(-x_max, x_max)
            
            # Add simple text labels with moderate spacing
            for i, (coef, p) in enumerate(zip(coefs, pvalues)):
                # Add asterisks for significance levels
                stars = ''
                if p < 0.001: stars = '***'
                elif p < 0.01: stars = '**'
                elif p < 0.05: stars = '*'
                
                # Always position text on the right side
                offset = 0.05  # Fixed positive offset to place text on the right
                text_pos = coef + offset
                
                # Short, concise text format
                text = f"{coef:.2f}{stars} (p={p:.3f})"
                
                # Always use left alignment
                ax.text(
                    text_pos, y_pos[i],
                    text,
                    verticalalignment='center',
                    horizontalalignment='left',  # Always left-aligned
                    fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7, pad=3, edgecolor='none')
                )
            
            # Add gridlines
            ax.grid(True, axis='x', linestyle=':', alpha=0.6)
        
        # Hide any unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        
        # Add legend for significance levels
        fig.text(0.98, 0.02, 
                '* p<0.05   ** p<0.01   *** p<0.001',
                fontsize=11,
                horizontalalignment='right')
        
        # Adjust layout with more space
        plt.tight_layout(h_pad=3, w_pad=3)
        
        # Save the plot
        file_name = os.path.join(self.output_dir, 'mixed_model_forest_plots.png')
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nPlot saved as: {file_name}")
    
    def plot_cognitive_trajectories(self):
        """Create line plots showing individual trajectories by menopausal status."""
        var_labels = {
            'TOTIDE_avg': 'Total IDE Score (averaged)',
            'NERVES_log': 'Nervousness Score',
            'SAD_sqrt': 'Sadness Score',
            'FEARFULA_sqrt': 'Fearfulness Score'
        }
        
        # Create a figure with subplots
        n_vars = len(self.outcome_vars)
        n_cols = min(3, n_vars)
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows), squeeze=False)
        axes = axes.flatten()
        
        for idx, var in enumerate(self.outcome_vars):
            # Get a subset of data for visualization (limit to 100 subjects)
            subset_ids = np.random.choice(
                self.data['SWANID'].unique(), 
                min(100, len(self.data['SWANID'].unique())), 
                replace=False
            )
            plot_data = self.data[self.data['SWANID'].isin(subset_ids)]
            
            # Sort by SWANID and VISIT to ensure proper trajectory
            plot_data = plot_data.sort_values(['SWANID', 'VISIT'])
            
            # Create a spaghetti plot by SWANID, colored by STATUS_Label
            ax = axes[idx]
            
            # First plot average trajectories by status
            sns.lineplot(
                data=self.data,
                x='VISIT', 
                y=var,
                hue='STATUS_Label',
                estimator='mean',
                errorbar=('ci', 95),
                ax=ax,
                linewidth=3,
                palette='coolwarm'
            )
            
            # Add individual trajectories with low alpha for a subset
            for swanid, group in plot_data.groupby('SWANID'):
                ax.plot(
                    group['VISIT'], 
                    group[var], 
                    alpha=0.1, 
                    color='gray',
                    linewidth=0.5
                )
            
            # Customize the plot
            ax.set_title(f'Trajectories for {var_labels[var]}', pad=20)
            ax.set_xlabel('Visit')
            ax.set_ylabel(var_labels[var])
            
            # Add summary statistics as text
            summary_stats = self.data.groupby('STATUS_Label', observed=True)[var].agg([
                'count', 'mean', 'std', 'median'
            ]).round(2)
            
            # Add summary statistics to the side of the plot
            stat_text = "Summary by Status:\n"
            for status, stats in summary_stats.iterrows():
                stat_text += f"\n{status}:\n"
                stat_text += f"n={stats['count']}, mean={stats['mean']}\n"
                stat_text += f"median={stats['median']}, sd={stats['std']}\n"
            
            # Position the text box outside the plot
            text_box = ax.text(
                1.05,  # x position
                0.5,   # y position
                stat_text,
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.8),
                transform=ax.transAxes,  # Use axes coordinates
                verticalalignment='center'
            )
        
        # Remove any unused subplots
        for j in range(idx + 1, len(axes)):
            axes[j].set_visible(False)
        
        # Add a main title
        fig.suptitle(
            'Cognitive Measure Trajectories by Menopausal Status',
            y=1.02,
            fontsize=14
        )
        
        # Adjust layout and display
        plt.tight_layout()
        
        # Save the plot
        file_name = os.path.join(self.output_dir, 'menopause_trajectories.png')
        fig.savefig(
            file_name,
            dpi=300,
            bbox_inches='tight'
        )
        
        plt.close()
        print(f"\nPlot saved as: {file_name}")

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
            
            print("\nPlotting mixed model results...")
            self.plot_forest_plots()
            
            print("\nCreating cognitive trajectory plots...")
            self.plot_cognitive_trajectories()

            print("\nAnalysis complete.")

        finally:
            sys.stdout = output_capture.terminal
            output_capture.close()

if __name__ == "__main__":
    # Initialize analysis with your data file
    analysis = MenopauseCognitionAnalysis("processed_combined_data.csv")
    analysis.run_complete_analysis()