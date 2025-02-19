import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
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
        self.symptom_vars = ['HOTFLAS', 'NUMHOTF', 'BOTHOTF', 'NITESWE', 'NUMNITS', 'BOTNITS',
                            'COLDSWE', 'NUMCLDS', 'BOTCLDS', 'STIFF', 'IRRITAB', 'MOODCHG']
        self.outcome_vars = ['TOTIDE1', 'TOTIDE2']
        self.control_vars = ['STATUS', 'LANGCOG']
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
            'TOTIDE1': 'Total IDE Score 1',
            'TOTIDE2': 'Total IDE Score 2'
        }
        self.output_dir = get_output_dir('2_symptoms_model', 'cross-sectional') 
        
    def prepare_data(self):
        """Prepare data including menopausal status categorization."""
        self.data['STATUS'] = pd.to_numeric(self.data['STATUS'], errors='coerce')
        # Keep only statuses of interest (surgical: 1, 8; natural: 2,3,4,5)
        self.data = self.data[self.data['STATUS'].isin([1, 2, 3, 4, 5, 8])]
        
        # Convert variables to numeric
        all_vars = self.symptom_vars + self.outcome_vars + self.control_vars + ['SWANID', 'VISIT']
        for var in all_vars:
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
        
        # If you still need an ordering for the natural stages, you can keep the categorical for STATUS_Label.
        # For instance, you may want natural stages to appear in order and have “Surgical” as a separate category.
        natural_order = ['Pre-menopause', 'Early Peri', 'Late Peri', 'Post-menopause']
        self.data['STATUS_Label'] = pd.Categorical(
            self.data['STATUS_Label'],
            categories=['Surgical'] + natural_order,
            ordered=True
        )
        
        # Standardize symptom variables
        self.data[self.symptom_vars] = (self.data[self.symptom_vars] - 
                                       self.data[self.symptom_vars].mean()) / self.data[self.symptom_vars].std()
        
        # Drop rows with missing values
        self.data = self.data.dropna(subset=all_vars)
    
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

        print("Data shape after preprocessing:", self.data.shape)
        return self.data

    def validate_data(self):
        """Perform comprehensive data validation."""
        variables_to_check = self.symptom_vars + self.outcome_vars + self.control_vars
        # Only validate the specified variables
        validator = DataValidator(
            data=self.data,
            variables=variables_to_check,
            output_dir=self.output_dir
        )
        
        # Run validation checks
        validation_results = validator.run_checks(
            checks=['missing', 'distributions', 'group_sizes', 'homogeneity', 'multicollinearity', 'independence'],
            grouping_var='STATUS_Label'
        )
        return validation_results
    
    def calculate_model_metrics(self, model_results):
        """
        Calculate AIC, BIC, and other fit metrics for mixed models.
        """
        # Get number of observations and parameters
        n = model_results.nobs
        k = len(model_results.params)  # Fixed effects
        k += len(model_results.random_effects)  # Random effects
        k += 1  # Residual variance
        
        # Calculate log-likelihood
        ll = model_results.llf
        
        # Calculate AIC and BIC
        aic = -2 * ll + 2 * k
        bic = -2 * ll + np.log(n) * k
        
        return {
            'AIC': aic,
            'BIC': bic,
            'Log_Likelihood': ll,
            'Num_Parameters': k,
            'Num_Observations': n
        }
    
    def fit_mixed_model(self, outcome_var, symptom_var):
        """Fit mixed-effects model for a specific symptom-outcome pair."""
        formula = f"{outcome_var} ~ {symptom_var} + {' + '.join(self.control_vars)}"
        
        model = MixedLM.from_formula(
            formula=formula,
            groups='SWANID',
            data=self.data
        )
        
        result = model.fit()

        # Calculate model metrics
        metrics = self.calculate_model_metrics(result)
        
        # Store metrics with the result
        result.model_metrics = metrics

        return result
    
    def plot_marginal_effects(self, outcome_vars=None, symptom_vars=None, n_cols=3):
        """Create marginal effects plots for symptom-cognition relationships."""
        if outcome_vars is None:
            outcome_vars = self.outcome_vars
        if symptom_vars is None:
            symptom_vars = self.symptom_vars
            
        for outcome_var in outcome_vars:
            n_vars = len(symptom_vars)
            n_rows = (n_vars + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, 
                                   figsize=(6*n_cols, 5*n_rows))
            if n_rows == 1:
                axes = np.array([axes])
            axes = axes.flatten()
            
            for i, symptom_var in enumerate(symptom_vars):
                print(f"\nFitting model for {symptom_var} → {outcome_var}")
                
                # Fit mixed model
                model = self.fit_mixed_model(outcome_var, symptom_var)
                
                # Create range of symptom values for prediction
                x_range = np.linspace(
                    self.data[symptom_var].min(),
                    self.data[symptom_var].max(),
                    100
                )
                
                # Create prediction data with mean values for control variables
                pred_data = pd.DataFrame({
                    symptom_var: x_range,
                    'STATUS': [self.data['STATUS'].mean()] * len(x_range),
                    'LANGCOG': [self.data['LANGCOG'].mean()] * len(x_range)
                })
                
                # Get predictions
                predictions = model.predict(pred_data)
                
                # Calculate confidence intervals using the variance of the coefficient
                cov_matrix = model.cov_params()
                var_coef = cov_matrix.loc[symptom_var, symptom_var]
                se = np.sqrt(var_coef * (x_range ** 2))
                ci_lower = predictions - 1.96 * se
                ci_upper = predictions + 1.96 * se
                
                # Plot
                axes[i].plot(x_range, predictions, 'b-', label='Predicted')
                axes[i].fill_between(x_range, ci_lower, ci_upper,
                                   alpha=0.2, color='b', label='95% CI')
                
                # Add scatter plot of actual data
                axes[i].scatter(self.data[symptom_var], 
                              self.data[outcome_var], 
                              alpha=0.1, color='gray', 
                              s=20)
                
                # Add correlation coefficient and p-value
                corr, p_val = stats.pearsonr(
                    self.data[symptom_var],
                    self.data[outcome_var]
                )
                axes[i].text(0.05, 0.95,
                           f'r = {corr:.3f}\np = {p_val:.3e}',
                           transform=axes[i].transAxes,
                           bbox=dict(facecolor='white', alpha=0.8))
                
                # Formatting
                axes[i].set_title(f'{self.var_labels[symptom_var]}')
                axes[i].set_xlabel('Standardized Symptom Score')
                axes[i].set_ylabel(f'{self.var_labels[outcome_var]}')
                axes[i].grid(True, linestyle='--', alpha=0.3)
                
                # Add coefficient and p-value from mixed model
                coef = model.params[symptom_var]
                p_val = model.pvalues[symptom_var]
                axes[i].text(0.05, 0.85,
                           f'β = {coef:.3f}\np = {p_val:.3e}',
                           transform=axes[i].transAxes,
                           bbox=dict(facecolor='white', alpha=0.8))
            
            # Remove empty subplots
            for j in range(i + 1, len(axes)):
                axes[j].remove()
            
            plt.suptitle(f'Marginal Effects on {self.var_labels[outcome_var]}',
                        y=1.02, fontsize=16)
            plt.tight_layout()
            
            file_name = f'{self.output_dir}/{outcome_var}_marginal_effects.png'
            plt.savefig(file_name)
            plt.close()
            
            # Print model summaries
            for symptom_var in symptom_vars:
                model = self.fit_mixed_model(outcome_var, symptom_var)
                print(f"\nModel summary for {symptom_var} → {outcome_var}:")
                print(model.summary().tables[1])

    def plot_mean_trajectories(self, outcome_vars=None, ci=0.95):
        """
        Plot mean trajectories with confidence bands for each outcome variable.
        
        Parameters:
            outcome_vars (list): List of outcome variables to plot. If None, uses all outcome variables.
            ci (float): Confidence interval level (0-1).
        """
        if outcome_vars is None:
            outcome_vars = self.outcome_vars
            
        for outcome_var in outcome_vars:
            plt.figure(figsize=(10, 6))
            
            # Calculate mean and confidence intervals
            grouped = self.data.groupby('SWANID')[outcome_var]
            means = grouped.mean()
            sems = grouped.std() / np.sqrt(grouped.count())
            ci_factor = stats.norm.ppf((1 + ci) / 2)
            ci_lower = means - ci_factor * sems
            ci_upper = means + ci_factor * sems
            
            # Plot mean trajectory
            time_points = np.arange(len(means))
            plt.plot(time_points, means, 'b-', label='Mean')
            plt.fill_between(time_points, ci_lower, ci_upper,
                            alpha=0.2, color='b', label=f'{ci*100}% CI')
            
            plt.title(f'Mean Trajectory for {self.var_labels[outcome_var]}')
            plt.xlabel('Time Point')
            plt.ylabel(outcome_var)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.3)
            
            file_name = f'{self.output_dir}/{outcome_var}_mean_trajectory.png'
            plt.savefig(file_name)
            plt.close()

    def plot_group_comparisons(self, outcome_vars=None, symptom_vars=None, n_groups=2):
        """
        Plot comparisons between symptom severity groups.
        
        Parameters:
            outcome_vars (list): List of outcome variables to plot
            symptom_vars (list): List of symptom variables to use for grouping
            n_groups (int): Number of groups to split the data into
        """
        if outcome_vars is None:
            outcome_vars = self.outcome_vars
        if symptom_vars is None:
            symptom_vars = self.symptom_vars
            
        for outcome_var in outcome_vars:
            for symptom_var in symptom_vars:
                try:
                    plt.figure(figsize=(12, 6))
                    
                    # Create symptom severity groups with proper handling of duplicates
                    try:
                        # First try qcut with duplicate handling
                        self.data['severity_group'] = pd.qcut(
                            self.data[symptom_var],
                            q=n_groups,
                            labels=[f'Group {i+1}' for i in range(n_groups)],
                            duplicates='drop'
                        )
                    except ValueError as e:
                        print(f"\nWarning: Could not create {n_groups} equal-sized groups for {symptom_var}.")
                        print("Falling back to cut method with equal intervals.")
                        
                        # Fallback to cut method
                        self.data['severity_group'] = pd.cut(
                            self.data[symptom_var],
                            bins=n_groups,
                            labels=[f'Group {i+1}' for i in range(n_groups)]
                        )
                    
                    # Get group statistics
                    group_stats = self.data.groupby('severity_group')[symptom_var].agg(['count', 'mean', 'std'])
                    print(f"\nGroup statistics for {symptom_var}:")
                    print(group_stats)
                    
                    # Plot trajectories for each group
                    for group in self.data['severity_group'].unique():
                        group_data = self.data[self.data['severity_group'] == group]
                        
                        # Calculate mean trajectory
                        mean_trajectory = group_data.groupby('VISIT')[outcome_var].mean()
                        
                        # Calculate standard error
                        counts = group_data.groupby('VISIT')[outcome_var].count()
                        std = group_data.groupby('VISIT')[outcome_var].std()
                        sem = std / np.sqrt(counts)
                        
                        # Create time points array
                        time_points = np.array(mean_trajectory.index)
                        
                        # Plot mean line
                        plt.plot(time_points, mean_trajectory.values, '-o', 
                                label=f'{group} (n={len(group_data.SWANID.unique())})')
                        
                        # Add confidence intervals
                        plt.fill_between(
                            time_points,
                            mean_trajectory - 1.96 * sem,
                            mean_trajectory + 1.96 * sem,
                            alpha=0.2
                        )
                    
                    # Customize plot
                    plt.title(f'{self.var_labels[outcome_var]} by {self.var_labels[symptom_var]} Severity Groups')
                    plt.xlabel('Visit Number')
                    plt.ylabel(self.var_labels[outcome_var])
                    plt.legend(title='Severity Groups', bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.grid(True, linestyle='--', alpha=0.3)
                    
                    # Add text box with group information
                    text_content = "Group Information:\n"
                    for idx, row in group_stats.iterrows():
                        text_content += f"{idx}: n={int(row['count'])}, mean={row['mean']:.2f}\n"
                    
                    plt.text(1.05, 0.5, text_content,
                            transform=plt.gca().transAxes,
                            bbox=dict(facecolor='white', alpha=0.8),
                            verticalalignment='center')
                    
                    # Adjust layout and save
                    plt.tight_layout()
                    file_name = f'{self.output_dir}/{outcome_var}_{symptom_var}_group_comparison.png'
                    plt.savefig(file_name, bbox_inches='tight', dpi=300)
                    plt.close()
                    
                except Exception as e:
                    print(f"\nError creating group comparison plot for {outcome_var} ~ {symptom_var}: {str(e)}")
                    continue
                
                # Remove temporary column
                if 'severity_group' in self.data.columns:
                    self.data = self.data.drop('severity_group', axis=1)

    def plot_violin(self, outcome_vars=None, time_points=None):
        """
        Create violin plots for each outcome variable at each time point.
        
        Parameters:
            outcome_vars (list): List of outcome variables to plot.
            time_points (list): List of time points to include.
        """
        if outcome_vars is None:
            outcome_vars = self.outcome_vars
        
        for outcome_var in outcome_vars:
            # Get unique time points if not specified
            unique_times = self.data['SWANID'].unique() if time_points is None else time_points
            
            # Create figure
            plt.figure(figsize=(12, 6))
            
            # Prepare data for violin plot
            plot_data = []
            labels = []
            for time in unique_times:
                time_data = self.data[self.data['SWANID'] == time][outcome_var]
                plot_data.append(time_data)
                labels.append(f'Time {time}')
            
            # Create violin plot
            parts = plt.violinplot(plot_data, positions=range(len(unique_times)))
            
            # Customize violin plot appearance
            for pc in parts['bodies']:
                pc.set_facecolor('royalblue')
                pc.set_alpha(0.7)
            
            for partname in ['cbars', 'cmins', 'cmaxes']:
                parts[partname].set_color('black')
            
            # Add box plots inside violin plots
            plt.boxplot(plot_data, positions=range(len(unique_times)),
                    widths=0.1, showfliers=False,
                    medianprops=dict(color="white"))
            
            # Customize plot
            plt.title(f'Distribution of {self.var_labels[outcome_var]} by Time Point')
            plt.xlabel('Time Point')
            plt.ylabel(self.var_labels[outcome_var])
            plt.xticks(range(len(unique_times)), labels)
            plt.grid(True, linestyle='--', alpha=0.3)
            
            # Save plot
            file_name = f'{self.output_dir}/{outcome_var}_violin.png'
            plt.savefig(file_name)
            plt.close()
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up output capture
        output_capture = OutputCapture(self.output_dir)
        sys.stdout = output_capture
        
        try:
            print("Preparing data...")
            self.prepare_data()

            print("\nFitting mixed models...")
            for outcome_var in self.outcome_vars:
                for symptom_var in self.symptom_vars:
                    print(f"\nFitting model for {symptom_var} → {outcome_var}")
                    model = self.fit_mixed_model(outcome_var, symptom_var)
                    print(model.summary())
            
            print("\nGenerating marginal effects plots...")
            self.plot_marginal_effects()

            print("\nGenerating mean trajectories...")
            self.plot_mean_trajectories()
            
            print("\nGenerating group comparison plots...")
            self.plot_group_comparisons()
            
            print("\nGenerating violin plots...")
            self.plot_violin()

            print("\nAnalysis complete")

        finally:
            sys.stdout = output_capture.terminal
            output_capture.close()

if __name__ == "__main__":
    file_path = "processed_combined_data.csv"
    analysis = MenopauseCognitionAnalysis(file_path)
    analysis.run_complete_analysis()