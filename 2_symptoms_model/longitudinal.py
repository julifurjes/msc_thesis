import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
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
        self.symptom_vars = ['HOTFLAS', 'NUMHOTF', 'BOTHOTF', 'NITESWE', 'NUMNITS', 'BOTNITS',
                            'COLDSWE', 'NUMCLDS', 'BOTCLDS', 'STIFF', 'IRRITAB', 'MOODCHG']
        self.outcome_vars = ['TOTIDE1', 'TOTIDE2', 'NERVES', 'SAD', 'FEARFULA']
        self.transformed_symptom_vars = []  # Will be populated after transformations
        self.transformed_outcome_vars = []  # Will be populated after transformations
        self.control_vars = ['STATUS', 'LANGCOG', 'AGE']
        self.mixed_model_results = {}
        self.var_labels = {
            'HOTFLAS': 'Hot Flashes',
            'NUMHOTF': 'Number of Hot Flashes',
            'BOTHOTF': 'Bothersomeness of Hot Flashes',
            'NITESWE': 'Night Sweats',
            'NUMNITS': 'Number of Night Sweats',
            'BOTNITS': 'Bothersomeness of Night Sweats',
            'COLDSWE': 'Cold Sweats',
            'NUMCLDS': 'Number of Cold Sweats',
            'BOTCLDS': 'Bothersomeness of Cold Sweats',
            'STIFF': 'Stiffness',
            'IRRITAB': 'Irritability',
            'MOODCHG': 'Mood Changes',
            'TOTIDE_avg': 'Total IDE Score (averaged)',
            'NERVES': 'Nervousness Score',
            'SAD': 'Sadness Score',
            'FEARFULA': 'Fearfulness Score', 
        }
        self.output_dir = get_output_dir('2_symptoms_model', 'longitudinal')
        
    def prepare_data(self):
        """Prepare data including menopausal status categorization and transformations."""
        self.data['STATUS'] = pd.to_numeric(self.data['STATUS'], errors='coerce')
        # Keep only statuses of interest (surgical: 1, 8; natural: 2,3,4,5)
        self.data = self.data[self.data['STATUS'].isin([1, 2, 3, 4, 5, 8])]

        # Convert variables to numeric
        all_vars = self.symptom_vars + self.outcome_vars + self.control_vars + ['SWANID', 'VISIT']
        for var in all_vars:
            if var in self.data.columns:
                self.data[var] = pd.to_numeric(self.data[var], errors='coerce')
        
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
        
        # If you still need an ordering for the natural stages, you can keep the categorical for STATUS_Label
        natural_order = ['Pre-menopause', 'Early Peri', 'Late Peri', 'Post-menopause']
        self.data['STATUS_Label'] = pd.Categorical(
            self.data['STATUS_Label'],
            categories=['Surgical'] + natural_order,
            ordered=True
        )
        
        # Standardize symptom variables
        self.data[self.symptom_vars] = (self.data[self.symptom_vars] - 
                                       self.data[self.symptom_vars].mean()) / self.data[self.symptom_vars].std()
        
        self.transform_variables()
    
    def transform_variables(self):
        """Apply transformations to address heteroscedasticity and multicollinearity for outcome variables."""
        # Variables needing log transformation (highest skewness)
        log_transform_vars = ['NERVES', 'NUMHOTF', 'NUMNITS', 'COLDSWE', 'NUMCLDS']
        
        # Variables needing sqrt transformation (moderately skewed)
        sqrt_transform_vars = ['FEARFULA', 'SAD', 'NITESWE', 'HOTFLAS', 'MOODCHG', 'IRRITAB']
        
        # Variables to keep as-is (low skewness)
        no_transform_vars = ['BOTHOTF', 'BOTNITS', 'BOTCLDS', 'STIFF']
        
        # Apply log transformations
        for var in log_transform_vars:
            if var in self.data.columns:
                self.data[f"{var}_log"] = np.log1p(self.data[var])
                self.transformed_symptom_vars.append(f"{var}_log")
                self.var_labels[f"{var}_log"] = f"{self.var_labels.get(var, var)} (Log)"
        
        # Apply sqrt transformations
        for var in sqrt_transform_vars:
            if var in self.data.columns:
                self.data[f"{var}_sqrt"] = np.sqrt(self.data[var])
                self.transformed_symptom_vars.append(f"{var}_sqrt")
                self.var_labels[f"{var}_sqrt"] = f"{self.var_labels.get(var, var)} (Sqrt)"
        
        # Add untransformed variables
        for var in no_transform_vars:
            if var in self.data.columns:
                self.transformed_symptom_vars.append(var)
        
        # 2. TRANSFORM OUTCOME VARIABLES
        
        # Create TOTIDE average
        if 'TOTIDE1' in self.data.columns and 'TOTIDE2' in self.data.columns:
            self.data['TOTIDE_avg'] = (self.data['TOTIDE1'] + self.data['TOTIDE2']) / 2
            self.var_labels['TOTIDE_avg'] = 'Total IDE Score (averaged)'
        
        # Apply transformations to other cognitive variables
        self.transformed_outcome_vars = []
        
        # Add TOTIDE_avg to transformed outcomes
        if 'TOTIDE_avg' in self.data.columns:
            self.transformed_outcome_vars.append('TOTIDE_avg')

    def run_mixed_models(self):
        """
        Run linear mixed-effects models for symptom-cognition relationships.
        Uses transformed outcome variables and includes AGE and VISIT in the models.
        Also includes weighting for unbalanced data.
        """
        print("\nRunning linear mixed-effects models with transformed variables...")
        
        # Loop through all transformed outcome variables
        for transformed_outcome in self.transformed_outcome_vars:
            print(f"Analyzing models for {self.var_labels.get(transformed_outcome, transformed_outcome)}")
            
            # Get the original variable name (for results dictionary keys)
            if transformed_outcome.endswith('_log'):
                original_outcome = transformed_outcome.replace('_log', '')
            elif transformed_outcome.endswith('_sqrt'):
                original_outcome = transformed_outcome.replace('_sqrt', '')
            elif transformed_outcome.endswith('_avg'):
                original_outcome = 'TOTIDE'  # Special case for average
            else:
                original_outcome = transformed_outcome
            
            # Now loop through each symptom
            for symptom in self.symptom_vars:
                # Create formula with status, symptom, LANGCOG, AGE, and VISIT
                formula = (f"{transformed_outcome} ~ {symptom} + C(STATUS_Label, Treatment('Pre-menopause')) + "
                          f"LANGCOG + AGE + VISIT")
                
                try:
                    # Drop rows with missing values for the current variables
                    analysis_data = self.data.dropna(subset=[transformed_outcome, symptom, 'STATUS_Label', 
                                                             'LANGCOG', 'AGE', 'VISIT'])
                    
                    # Reset index to avoid potential index issues
                    analysis_data = analysis_data.reset_index(drop=True)
                    
                    # Add weights to handle unbalanced data (from first model)
                    status_counts = analysis_data['STATUS_Label'].value_counts()
                    analysis_data['weights'] = analysis_data['STATUS_Label'].map(
                        lambda x: 1 / (status_counts[x] / sum(status_counts))
                    )
                    
                    # Fit mixed model with random intercept for SWANID and random slope for VISIT (from first model)
                    model = mixedlm(
                        formula=formula,
                        groups=analysis_data["SWANID"],
                        data=analysis_data,
                        re_formula="~VISIT"  # Random slope for VISIT (from first model)
                    )
                    
                    # Add weights and fit the model
                    results = model.fit(reml=True, weights=analysis_data['weights'])
                    key = f"{original_outcome}_{symptom}"
                    self.mixed_model_results[key] = results
                    
                    # Print detailed results
                    print(f"\nMixed Model Results for {self.var_labels.get(symptom, symptom)} → "
                          f"{self.var_labels.get(original_outcome, original_outcome)}")
                    print(f"Using transformed variable: {transformed_outcome}")
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
                    
                    # Check model diagnostics
                    self.check_model_diagnostics(results, original_outcome, transformed_outcome, symptom, analysis_data)
                    
                except Exception as e:
                    print(f"Error in mixed model analysis for {symptom} → {original_outcome}: {str(e)}")
    
    def check_model_diagnostics(self, model_results, outcome, transformed_outcome, symptom, data):
        """Check model residuals and diagnostics for the mixed model."""
        try:
            # Calculate residuals
            predicted = model_results.predict(data)
            actual = data[transformed_outcome]
            residuals = actual - predicted
            
            # Create directory for diagnostics
            diag_dir = os.path.join(self.output_dir, "model_diagnostics")
            os.makedirs(diag_dir, exist_ok=True)
            
            # Plot residuals
            plt.figure(figsize=(12, 10))
            
            plt.subplot(2, 2, 1)
            plt.scatter(predicted, residuals, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='-')
            title_text = f'Residuals vs Fitted for {self.var_labels.get(outcome, outcome)} ~ {self.var_labels.get(symptom, symptom)}'
            plt.title(title_text)
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
            plt.savefig(os.path.join(diag_dir, f'{outcome}_{symptom}_diagnostics.png'))
            plt.close()
            
            # Check normality of residuals using different methods 
            # Depending on sample size
            if len(residuals) > 5000:
                # For large samples, use Anderson-Darling test with a sample
                sample_size = min(5000, len(residuals))
                residual_sample = pd.Series(residuals).sample(sample_size)
                _, ad_p = sm.stats.diagnostic.normal_ad(residual_sample.dropna())
                print(f"\nAnderson-Darling normality test p-value (on {sample_size} sampled residuals): {ad_p:.4f}")
                
                # Also try Shapiro-Wilk on a smaller sample for comparison
                shapiro_sample = pd.Series(residuals).sample(1000)  # Shapiro-Wilk works best with smaller samples
                _, sw_p = stats.shapiro(shapiro_sample.dropna())
                print(f"Shapiro-Wilk normality test p-value (on 1000 sampled residuals): {sw_p:.4f}")
            else:
                # For smaller samples, use both tests on the full data
                _, ad_p = sm.stats.diagnostic.normal_ad(residuals.dropna())
                _, sw_p = stats.shapiro(residuals.dropna())
                print(f"\nAnderson-Darling normality test p-value: {ad_p:.4f}")
                print(f"Shapiro-Wilk normality test p-value: {sw_p:.4f}")
            
            # Decide if normality is violated based on both tests
            if ad_p < 0.05 and sw_p < 0.05:
                print("WARNING: Residuals may not be normally distributed.")
                print("However, with this large sample size, the model is still robust to moderate non-normality.")
            else:
                print("Residuals appear to be more normally distributed after transformation.")
            
        except Exception as e:
            print(f"Error in model diagnostics: {str(e)}")
    
    def plot_symptom_effects(self):
        """
        Create forest plots showing the effects of symptoms on cognitive outcomes.
        """
        if not self.mixed_model_results:
            print("No mixed model results available. Run run_mixed_models() first.")
            return
        
        print("\nCreating forest plots for transformed models...")
        
        # Get unique outcomes from the keys
        unique_outcomes = set()
        for key in self.mixed_model_results.keys():
            outcome = key.split('_')[0]
            unique_outcomes.add(outcome)
        
        # Group results by outcome variable
        for outcome in unique_outcomes:
            # Get all models for this outcome
            outcome_models = {k: v for k, v in self.mixed_model_results.items() if k.startswith(f"{outcome}_")}
            
            if not outcome_models:
                print(f"No models found for {outcome}")
                continue
                
            print(f"Creating forest plot for {outcome} with {len(outcome_models)} symptoms")
            
            # Create figure for this outcome
            plt.figure(figsize=(12, 10))
            
            # Initialize lists to store plot data
            symptoms = []
            coefs = []
            errors = []
            pvalues = []
            
            # Extract coefficients for each symptom
            for key, results in outcome_models.items():
                symptom = key.split('_')[1]
                
                # Get coefficient for symptom (it's the first predictor after the intercept)
                param_name = symptom
                
                if param_name in results.params.index:
                    symptoms.append(self.var_labels.get(symptom, symptom))
                    coefs.append(results.params[param_name])
                    errors.append(results.bse[param_name])
                    pvalues.append(results.pvalues[param_name])
                else:
                    print(f"Warning: Coefficient for {symptom} not found in model results")
            
            if not symptoms:
                print(f"No valid coefficients found for {outcome}")
                continue
                
            # Sort by coefficient magnitude for better visualization
            sorted_indices = np.argsort(np.abs(coefs))[::-1]  # Sort by absolute value, descending
            symptoms = [symptoms[i] for i in sorted_indices]
            coefs = [coefs[i] for i in sorted_indices]
            errors = [errors[i] for i in sorted_indices]
            pvalues = [pvalues[i] for i in sorted_indices]
            
            # Create forest plot
            y_pos = np.arange(len(symptoms))
            
            plt.errorbar(
                coefs, y_pos,
                xerr=1.96 * np.array(errors),  # 95% CI
                fmt='o',
                capsize=5,
                markersize=8,
                elinewidth=2,
                capthick=2
            )
            
            # Add vertical line at zero
            plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            
            # Customize plot
            plt.yticks(y_pos, symptoms, fontsize=12)
            plt.title(f'Effects of Symptoms on {self.var_labels.get(outcome, outcome)} (Transformed)', fontsize=14)
            plt.xlabel('Standardized Coefficient (95% CI)', fontsize=12)
            
            # Add text labels with p-values
            for i, (coef, p) in enumerate(zip(coefs, pvalues)):
                # Add asterisks for significance levels
                stars = ''
                if p < 0.001: stars = '***'
                elif p < 0.01: stars = '**'
                elif p < 0.05: stars = '*'
                
                # Position text based on coefficient sign
                text_pos = coef + 0.02 * np.sign(coef) * max(abs(np.array(coefs)))
                ha = 'left' if coef >= 0 else 'right'
                
                plt.text(
                    text_pos, y_pos[i],
                    f"{coef:.3f}{stars} (p={p:.3f})",
                    verticalalignment='center',
                    horizontalalignment=ha,
                    fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7, pad=3, edgecolor='none')
                )
            
            # Add gridlines
            plt.grid(True, axis='x', linestyle=':', alpha=0.6)
            
            # Add legend for significance levels
            plt.figtext(0.98, 0.02, 
                    '* p<0.05   ** p<0.01   *** p<0.001',
                    fontsize=11,
                    horizontalalignment='right')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save the plot
            file_name = os.path.join(self.output_dir, f'{outcome}_symptom_effects.png')
            plt.savefig(file_name, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Forest plot saved as: {file_name}")
    
    def plot_trajectory_patterns(self):
        """
        Create line plots showing trajectories of symptoms and cognitive measures across visits,
        grouped by menopausal status.
        """
        # Variables to plot (original variables, not transformed)
        all_vars_to_plot = self.symptom_vars + self.outcome_vars
        
        # Create figures with 3 variables per row
        n_vars = len(all_vars_to_plot)
        n_cols = 3
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows), squeeze=False)
        axes = axes.flatten()
        
        for idx, var in enumerate(all_vars_to_plot):
            # Get a subset of data for individual trajectories (100 random subjects)
            subset_ids = np.random.choice(
                self.data['SWANID'].unique(), 
                min(100, len(self.data['SWANID'].unique())), 
                replace=False
            )
            plot_data = self.data[self.data['SWANID'].isin(subset_ids)]
            
            # Sort by SWANID and VISIT to ensure proper trajectory
            plot_data = plot_data.sort_values(['SWANID', 'VISIT'])
            
            # Create a spaghetti plot
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
            ax.set_title(f'Trajectories for {self.var_labels.get(var, var)}', pad=20)
            ax.set_xlabel('Visit')
            ax.set_ylabel(self.var_labels.get(var, var))
            
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
            'Variable Trajectories by Menopausal Status',
            y=1.02,
            fontsize=14
        )
        
        # Adjust layout and display
        plt.tight_layout()
        
        # Save the plot
        file_name = os.path.join(self.output_dir, 'all_variables_trajectories.png')
        fig.savefig(
            file_name,
            dpi=300,
            bbox_inches='tight'
        )
        
        plt.close()
        print(f"\nTrajectory plot saved as: {file_name}")

    def run_complete_analysis(self):
        """Run the complete analysis pipeline with transformed variables."""
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up output capture
        output_capture = OutputCapture(self.output_dir)
        sys.stdout = output_capture
        
        try:
            print("\nRunning full analysis with transformations from both models...")
            self.prepare_data()
            
            print("\nRunning mixed models analysis with transformed variables...")
            self.run_mixed_models()
            
            print("\nCreating symptom effects plots...")
            self.plot_symptom_effects()
            
            print("\nCreating trajectory pattern plots...")
            self.plot_trajectory_patterns()

            print("\nAnalysis complete")

        finally:
            sys.stdout = output_capture.terminal
            output_capture.close()

if __name__ == "__main__":
    file_path = "processed_combined_data.csv"
    analysis = MenopauseCognitionAnalysis(file_path)
    analysis.run_complete_analysis()