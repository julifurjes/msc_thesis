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

class MenopauseCognitionAnalysis:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path, low_memory=False)
        self.symptom_vars = ['HOTFLAS', 'NUMHOTF', 'BOTHOTF', 'NITESWE', 'NUMNITS', 'BOTNITS',
                            'COLDSWE', 'NUMCLDS', 'BOTCLDS', 'STIFF', 'IRRITAB', 'MOODCHG']
        self.outcome_vars = ['TOTIDE1', 'TOTIDE2', 'NERVES', 'SAD', 'FEARFULA']
        self.transformed_symptom_vars = []  # Will be populated after transformations
        self.transformed_outcome_vars = []  # Will be populated after transformations
        self.control_vars = ['STATUS', 'AGE']
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
        
        natural_order = ['Pre-menopause', 'Early Peri', 'Late Peri', 'Post-menopause']
        self.data['STATUS_Label'] = pd.Categorical(
            self.data['STATUS_Label'],
            categories=['Surgical'] + natural_order,
            ordered=True
        )
        
        # Standardize symptom variables
        self.data[self.symptom_vars] = (self.data[self.symptom_vars] - 
                                       self.data[self.symptom_vars].mean()) / self.data[self.symptom_vars].std()
        
        # Make langauge categorical
        self.data['LANGCOG'] = self.data['LANGCOG'].astype('category')
        
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

        # Get the reference language for the model (the most common one)
        reference_language = self.data['LANGCOG'].mode()[0]
        
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
            
            # Loop through each symptom
            for symptom in self.symptom_vars:
                # Create formula with status, symptom, AGE, and VISIT
                formula = (f"{transformed_outcome} ~ {symptom} + C(STATUS_Label, Treatment('Pre-menopause')) + "
                          f"AGE + VISIT + C(LANGCOG, Treatment({reference_language}))")
                
                try:
                    # Drop rows with missing values for the current variables
                    analysis_data = self.data.dropna(subset=[transformed_outcome, symptom, 'STATUS_Label', 'AGE', 'VISIT', 'LANGCOG'])
                    
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
        
        # Import the palette and select distinct colors
        green_palette = sns.color_palette("YlGn", n_colors=8)
        # Create a larger palette to select from
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
                return 'gray', ''                 # Not significant: gray
        
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
            plt.figure(figsize=(12, 16))
            
            # Initialize lists to store plot data
            symptoms = []
            coefs = []
            errors = []
            pvalues = []
            colors = []
            stars = []
            
            # Extract coefficients for each symptom
            for key, results in outcome_models.items():
                symptom = key.split('_')[1]
                
                # Get coefficient for symptom (it's the first predictor after the intercept)
                param_name = symptom
                
                if param_name in results.params.index:
                    p_value = results.pvalues[param_name]
                    color, star = get_color(p_value)
                    
                    symptoms.append(self.var_labels.get(symptom, symptom))
                    coefs.append(results.params[param_name])
                    errors.append(results.bse[param_name])
                    pvalues.append(p_value)
                    colors.append(color)
                    stars.append(star)
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
            colors = [colors[i] for i in sorted_indices]
            stars = [stars[i] for i in sorted_indices]
            
            # Create forest plot with color-coded points
            y_pos = np.arange(len(symptoms))
            
            # Plot each point individually with its own color
            for i, (coef, error, color) in enumerate(zip(coefs, errors, colors)):
                plt.errorbar(
                    coef, y_pos[i],
                    xerr=1.96 * error,  # 95% CI
                    fmt='o',
                    color=color,
                    capsize=5,
                    markersize=8,
                    elinewidth=2,
                    capthick=2
                )
            
            # Add vertical line at zero
            plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            
            # Customize plot
            plt.yticks(y_pos, symptoms, fontsize=18)
            plt.xticks(fontsize=18)
            
            # Calculate reasonable x-axis limits based on coefficients and errors
            all_values = []
            for coef, error in zip(coefs, errors):
                all_values.extend([coef - 1.96 * error, coef + 1.96 * error])
            
            x_min, x_max = min(all_values), max(all_values)
            x_range = x_max - x_min
            plt.xlim(x_min - 0.1 * x_range, x_max + 0.3 * x_range)  # Extra space on right for labels
            
            # Add text labels with p-values - aligned to the right
            for i, (coef, error, p, star, color) in enumerate(zip(coefs, errors, pvalues, stars, colors)):
                # Calculate the right edge of the error bar
                upper_error = coef + 1.96 * error
                
                # Position label to the right of the error bar with some padding
                label_x = upper_error + 0.02 * x_range
                
                plt.text(
                    label_x, y_pos[i],
                    f'{coef:.3f} {star}',
                    verticalalignment='center',
                    horizontalalignment='left',
                    fontsize=18,
                    color=color,
                    fontweight='bold' if p < 0.05 else 'normal',
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2', edgecolor='none')
                )
            
            # Add gridlines
            plt.grid(True, axis='x', linestyle=':', alpha=0.6)
            
            # Create custom legend for significance levels and colors
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=selected_greens[3], label='p < 0.001 (***)', edgecolor='black'),
                Patch(facecolor=selected_greens[2], label='p < 0.01 (**)', edgecolor='black'),
                Patch(facecolor=selected_greens[1], label='p < 0.05 (*)', edgecolor='black'),
                Patch(facecolor='gray', label='p ≥ 0.05 (n.s.)', edgecolor='black')
            ]
            
            plt.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.98, 0.02))
            
            # Adjust layout
            plt.tight_layout()
            
            # Save the plot
            file_name = os.path.join(self.output_dir, f'{outcome}_forest_plot.png')
            plt.savefig(file_name, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Forest plot saved as: {file_name}")

    def analyze_symptom_intensity_by_stage(self):
        """
        Analyze how symptom intensity (frequency, daily count, bothersomeness) 
        varies across different menopausal stages, including surgical menopause.
        """
        print("\nAnalyzing symptom intensity variation across menopausal stages...")
        
        # Create output directory for these specific results
        intensity_dir = os.path.join(self.output_dir, "symptom_intensity")
        os.makedirs(intensity_dir, exist_ok=True)
        
        # Define the symptom groups to analyze
        symptom_groups = {
            'Hot Flashes': ['HOTFLAS', 'NUMHOTF', 'BOTHOTF'],
            'Night Sweats': ['NITESWE', 'NUMNITS', 'BOTNITS'],
            'Cold Sweats': ['COLDSWE', 'NUMCLDS', 'BOTCLDS'],
            'Mood': ['IRRITAB', 'MOODCHG'],
            'Stiffness': ['STIFF'],
        }
        
        # Status order for plotting
        status_order = ['Pre-menopause', 'Early Peri', 'Late Peri', 'Post-menopause', 'Surgical']
        
        # Initialize a DataFrame to store results
        results_data = []
        
        # Analyze each symptom group
        for symptom_group, symptom_vars in symptom_groups.items():
            print(f"\nAnalyzing {symptom_group} symptoms...")
            
            # Create figure for this symptom group
            fig, axes = plt.subplots(1, len(symptom_vars), figsize=(len(symptom_vars)*5, 6), squeeze=False)
            axes = axes.flatten()
            
            # For each measure of the symptom
            for i, symptom_var in enumerate(symptom_vars):
                if symptom_var not in self.data.columns:
                    print(f"Warning: {symptom_var} not found in data")
                    continue
                    
                # Compute summary statistics
                summary = self.data.groupby('STATUS_Label', observed=True)[symptom_var].agg([
                    'count', 'mean', 'std', 'median', 'min', 'max'
                ]).reset_index()
                
                # Convert summary to proper category type with correct order
                summary['STATUS_Label'] = pd.Categorical(
                    summary['STATUS_Label'],
                    categories=status_order,
                    ordered=True
                )
                
                # Sort by the ordered category
                summary = summary.sort_values('STATUS_Label')
                
                # Store results for later reporting
                for _, row in summary.iterrows():
                    results_data.append({
                        'Symptom Group': symptom_group,
                        'Symptom': self.var_labels.get(symptom_var, symptom_var),
                        'Menopausal Stage': row['STATUS_Label'],
                        'Count': row['count'],
                        'Mean': row['mean'],
                        'StdDev': row['std'],
                        'Median': row['median'],
                        'Min': row['min'],
                        'Max': row['max']
                    })
                
                # Create a bar plot
                ax = axes[i]
                
                # Calculate standard error for error bars
                summary['se'] = summary['std'] / np.sqrt(summary['count'])
                
                # Create the bar plot with error bars
                bars = ax.bar(
                    x=np.arange(len(summary)),
                    height=summary['mean'],
                    yerr=summary['se'],
                    capsize=4,
                    width=0.7,
                    color=sns.color_palette("YlGnBu", n_colors=len(summary))
                )
                
                # Add mean value labels on top of each bar
                for j, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width()/2.,
                        height + 0.1,
                        f'{summary["mean"].iloc[j]:.2f}',
                        ha='center', 
                        va='bottom',
                        fontsize=9,
                        rotation=0
                    )
                
                # Set axis labels and title
                ax.set_title(self.var_labels.get(symptom_var, symptom_var), fontsize=12)
                ax.set_ylabel('Mean Score (with SE)', fontsize=10)
                ax.set_xticks(np.arange(len(summary)))
                ax.set_xticklabels(summary['STATUS_Label'], rotation=45, ha='right', fontsize=9)
            
            # Add an overall title
            fig.suptitle(f'{symptom_group} Intensity by Menopausal Stage', fontsize=14, y=1.05)
            
            # Adjust layout to prevent overlap
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(
                os.path.join(intensity_dir, f'{symptom_group.lower().replace(" ", "_")}_intensity.png'),
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()
        
        # Create a comprehensive summary table as a DataFrame
        results_df = pd.DataFrame(results_data)
        
        # Create formatted tables for each symptom group
        for symptom_group in results_df['Symptom Group'].unique():
            group_data = results_df[results_df['Symptom Group'] == symptom_group]
            
            # Pivot the data to create a nicely formatted table
            pivot_mean = pd.pivot_table(
                group_data, 
                values='Mean', 
                index='Menopausal Stage',
                columns='Symptom',
                aggfunc='first'
            )
            
            # Add count information (average across symptoms)
            count_data = group_data.groupby('Menopausal Stage')['Count'].mean().astype(int)
            pivot_mean['Sample Size'] = count_data
            
            # Sort rows by menopausal stage order
            pivot_mean = pivot_mean.reindex(status_order)
            
            # Print the table
            print(f"\n{symptom_group} Intensity by Menopausal Stage:")
            print("=" * 80)
            print(pivot_mean.round(2).to_string())
            print("=" * 80)
            
            # Save the table to a CSV file
            pivot_mean.to_csv(
                os.path.join(intensity_dir, f'{symptom_group.lower().replace(" ", "_")}_intensity.csv')
            )
        
        # Plot overall symptom intensity pattern
        self.plot_overall_symptom_intensity_pattern(results_df, intensity_dir)
        
        print("Symptom intensity analysis complete. Results saved to:", intensity_dir)
        return results_df

    def plot_overall_symptom_intensity_pattern(self, results_df, output_dir):
        """
        Create a comprehensive visualization showing how symptom intensity 
        patterns change across menopausal stages.
        """
        # Status order
        status_order = ['Pre-menopause', 'Early Peri', 'Late Peri', 'Post-menopause', 'Surgical']
        
        # Create figure for the comprehensive visualization
        plt.figure(figsize=(14, 12))
        
        # Add a subplot for symptom intensity patterns
        plt.subplot(111)
        
        # Group by symptom group and menopausal stage, calculating mean intensity
        intensity_pattern = results_df.groupby(['Symptom Group', 'Menopausal Stage'])['Mean'].mean().reset_index()
        
        # Convert to proper categorical type
        intensity_pattern['Menopausal Stage'] = pd.Categorical(
            intensity_pattern['Menopausal Stage'],
            categories=status_order,
            ordered=True
        )
        
        # Sort by stage
        intensity_pattern = intensity_pattern.sort_values('Menopausal Stage')
        
        # Use the YlGn color palette
        green_palette = sns.color_palette("YlGn", n_colors=10)
        
        # Get unique symptom groups for custom color assignment
        symptom_groups = intensity_pattern['Symptom Group'].unique()
        
        # Select specific colors from the palette for each symptom group
        # Use indices that are well-spaced for visual distinction
        color_indices = [1, 3, 5, 7, 9][:len(symptom_groups)]
        custom_colors = [green_palette[i] for i in color_indices]
        
        # Create a color dictionary
        color_dict = dict(zip(symptom_groups, custom_colors))
        
        # Create line plot with custom colors
        sns.lineplot(
            data=intensity_pattern,
            x='Menopausal Stage',
            y='Mean',
            hue='Symptom Group',
            marker='o',
            markersize=10,
            linewidth=3,
            err_style='band',
            palette=color_dict
        )
        
        # Set title and labels
        plt.xlabel('Menopausal Stage', fontsize=18)
        plt.ylabel('Mean Intensity Score', fontsize=18)
        plt.xticks(rotation=45, ha='right', fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Enhance the legend
        plt.legend(
            title='Symptom Group',
            fontsize=18,
            title_fontsize=18,
            bbox_to_anchor=(1.05, 1),
            loc='upper left'
        )
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'overall_symptom_intensity_pattern.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_symptom_intensity_heatmap(self):
        """
        Create a heat map showing symptom intensity across menopausal stages.
        5 × 4 matrix: stage (rows) × symptom group (cols).
        Color represents z-scored intensity values.
        """
        print("\nCreating symptom intensity heat map...")
        
        # Define symptom groups (matching your analyze_symptom_intensity_by_stage method)
        symptom_groups = {
            'Hot Flashes': ['HOTFLAS', 'NUMHOTF', 'BOTHOTF'],
            'Night Sweats': ['NITESWE', 'NUMNITS', 'BOTNITS'], 
            'Cold Sweats': ['COLDSWE', 'NUMCLDS', 'BOTCLDS'],
            'Mood Symptoms': ['IRRITAB', 'MOODCHG'],
            'Stiffness': ['STIFF']
        }
        
        # Status order for the heat map (rows)
        status_order = ['Pre-menopause', 'Early Peri', 'Late Peri', 'Post-menopause', 'Surgical']
        
        # Initialize matrix to store z-scored intensities
        heatmap_data = pd.DataFrame(index=status_order, columns=list(symptom_groups.keys()))
        
        # Calculate mean intensity for each symptom group and menopausal stage
        for group_name, symptom_vars in symptom_groups.items():
            print(f"Processing {group_name}...")
            
            # Get available symptoms from this group
            available_symptoms = [var for var in symptom_vars if var in self.data.columns]
            
            if not available_symptoms:
                print(f"Warning: No symptoms found for {group_name}")
                continue
            
            # Calculate mean intensity across symptoms in this group for each stage
            group_means = []
            for status in status_order:
                # Get data for this status
                status_data = self.data[self.data['STATUS_Label'] == status]
                
                if len(status_data) == 0:
                    group_means.append(np.nan)
                    continue
                
                # Calculate mean across all symptoms in this group for this status
                symptom_means = []
                for symptom in available_symptoms:
                    symptom_mean = status_data[symptom].mean()
                    if not np.isnan(symptom_mean):
                        symptom_means.append(symptom_mean)
                
                if symptom_means:
                    group_mean = np.mean(symptom_means)
                    group_means.append(group_mean)
                else:
                    group_means.append(np.nan)
            
            # Store in heatmap data
            heatmap_data[group_name] = group_means
        
        # Convert to numeric and calculate z-scores
        heatmap_data = heatmap_data.astype(float)
        
        # Calculate z-scores for each symptom group (column-wise standardization)
        heatmap_data_z = heatmap_data.copy()
        for col in heatmap_data.columns:
            col_data = heatmap_data[col].dropna()
            if len(col_data) > 1:
                mean_val = col_data.mean()
                std_val = col_data.std()
                if std_val > 0:
                    heatmap_data_z[col] = (heatmap_data[col] - mean_val) / std_val
                else:
                    heatmap_data_z[col] = 0
        
        print("Z-scored intensity matrix:")
        print(heatmap_data_z.round(2))
        
        # Create custom green palette
        green_palette = sns.color_palette("YlGn", n_colors=8)
        
        # Create the Z-SCORED heat map
        plt.figure(figsize=(12, 14))
        
        sns.heatmap(
            heatmap_data_z,
            annot=True,  # Show values in cells
            fmt='.2f',   # Format to 2 decimal places
            cmap=green_palette,
            square=True, # Make cells square
            linewidths=0.5,
            cbar_kws={
                'label': 'Z-scored Symptom Intensity',
                'shrink': 0.8
            },
            annot_kws={'size': 18, 'weight': 'bold'}
        )
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right', fontsize=18)
        plt.yticks(rotation=0, fontsize=18)
        
        # Add a subtle grid
        plt.grid(False)  # Remove default grid from heatmap
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        file_name = os.path.join(self.output_dir, 'symptom_intensity_heatmap_zscore.png')
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Z-scored symptom intensity heat map saved as: {file_name}")
        
        # Create the RAW VALUES heat map
        plt.figure(figsize=(12, 8))
        
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.3f',
            cmap=green_palette,
            square=True,
            linewidths=0.5,
            cbar_kws={
                'label': 'Mean Symptom Intensity',
                'shrink': 0.8
            },
            annot_kws={'size': 12, 'weight': 'bold'}
        )
        
        plt.title('Symptom Intensity Heat Map Across Menopausal Stages', 
                fontsize=16, pad=20)
        plt.xlabel('Symptom Groups', fontsize=14)
        plt.ylabel('Menopausal Stage', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # Save the raw values version
        file_name_raw = os.path.join(self.output_dir, 'symptom_intensity_heatmap_raw.png')
        plt.savefig(file_name_raw, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Raw values heat map saved as: {file_name_raw}")
        
        # Return the data for further analysis if needed
        return heatmap_data_z, heatmap_data

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

            print("\nAnalyzing symptom intensity by menopausal stage...")
            self.analyze_symptom_intensity_by_stage()

            print("\nCreating symptom intensity heat map...")
            self.plot_symptom_intensity_heatmap()

            print("\nAnalysis complete")

        finally:
            sys.stdout = output_capture.terminal
            output_capture.close()

if __name__ == "__main__":
    file_path = "processed_combined_data.csv"
    analysis = MenopauseCognitionAnalysis(file_path)
    analysis.run_complete_analysis()