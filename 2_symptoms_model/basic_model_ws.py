import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.formula.api import gee
from statsmodels.genmod.cov_struct import Exchangeable
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
        self.outcome_vars = ['TOTIDE1', 'TOTIDE2']
        self.control_vars = ['STATUS', 'LANGCOG']
        self.gee_results = {}
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
        self.output_dir = get_output_dir('2_symptoms_model', 'within-subjects') 
        
    def prepare_data(self):
        """Prepare data including menopausal status categorization."""
        # Convert variables to numeric
        all_vars = self.symptom_vars + self.outcome_vars + self.control_vars + ['SWANID', 'VISIT']
        for var in all_vars:
            self.data[var] = pd.to_numeric(self.data[var], errors='coerce')
        
        # Map STATUS to labels
        status_map = {
            2: 'Post-menopause',
            3: 'Late Peri',
            4: 'Early Peri',
            5: 'Pre-menopause'
        }
        self.data['STATUS_Label'] = self.data['STATUS'].map(status_map)
        
        # Create ordered categorical for status
        status_order = ['Pre-menopause', 'Early Peri', 'Late Peri', 'Post-menopause']
        self.data['STATUS_Label'] = pd.Categorical(
            self.data['STATUS_Label'],
            categories=status_order,
            ordered=True
        )
        
        # Standardize symptom variables
        self.data[self.symptom_vars] = (self.data[self.symptom_vars] - 
                                       self.data[self.symptom_vars].mean()) / self.data[self.symptom_vars].std()
        
        # Drop rows with missing values
        self.data = self.data.dropna(subset=all_vars)
        
        print("Data shape after preprocessing:", self.data.shape)
        return self.data

    def calculate_baseline_changes(self):
        """Calculate changes from baseline for each subject."""
        # Sort by subject ID and visit
        self.data = self.data.sort_values(['SWANID', 'VISIT'])
        
        # For each subject, use their Pre-menopause measurement as baseline
        baseline_data = (self.data[self.data['STATUS_Label'] == 'Pre-menopause']
                        .groupby('SWANID')
                        .first()
                        .reset_index())
        
        # Calculate baseline changes for outcomes and symptoms
        vars_to_change = self.outcome_vars + self.symptom_vars
        baseline_cols = {col: f'{col}_baseline' for col in vars_to_change}
        baseline_data = baseline_data[['SWANID'] + vars_to_change].rename(columns=baseline_cols)
        
        # Merge baseline data back
        self.data = self.data.merge(baseline_data, on='SWANID', how='left')
        
        # Calculate changes from baseline
        for var in vars_to_change:
            self.data[f'{var}_change'] = self.data[var] - self.data[f'{var}_baseline']
            self.data[f'{var}_pct_change'] = (
                (self.data[var] - self.data[f'{var}_baseline']) / 
                self.data[f'{var}_baseline'] * 100
            )
        
        return self.data

    def run_gee_analysis(self, outcome_vars=None, symptom_vars=None):
        """Run GEE analysis for symptom-cognition relationships."""
        if outcome_vars is None:
            outcome_vars = self.outcome_vars
        if symptom_vars is None:
            symptom_vars = self.symptom_vars
            
        for outcome in outcome_vars:
            for symptom in symptom_vars:
                # Create formula with status and symptom
                formula = (f"{outcome} ~ {symptom} + C(STATUS_Label, Treatment('Pre-menopause'))")
                
                try:
                    # Fit GEE model
                    model = gee(
                        formula=formula,
                        groups="SWANID",
                        data=self.data.dropna(subset=[outcome, symptom]),
                        cov_struct=Exchangeable(),
                        family=sm.families.Gaussian()
                    )
                    
                    results = model.fit()
                    key = f"{outcome}_{symptom}"
                    self.gee_results[key] = results
                    
                    # Print results
                    print(f"\nGEE Results for {self.var_labels[symptom]} → {self.var_labels[outcome]}")
                    print("=" * 50)
                    print(results.summary())
                    
                except Exception as e:
                    print(f"Error in GEE analysis for {symptom} → {outcome}: {str(e)}")

    def plot_change_distributions(self, use_percentage=False):
        """Plot distributions of changes across menopausal stages and print statistics."""
        # Print header for terminal output
        print("\n" + "="*80)
        print(f"{'ABSOLUTE' if not use_percentage else 'PERCENTAGE'} CHANGE STATISTICS")
        print("="*80)

        # Create separate figures for outcomes and symptoms
        for var_set, title in [(self.outcome_vars, 'Cognitive Measures'),
                             (self.symptom_vars, 'Symptoms')]:
            
            print(f"\n{title.upper()}:")
            print("-"*80)
            
            n_vars = len(var_set)
            n_cols = 3
            n_rows = (n_vars + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            # Convert axes to a flattened array
            if n_rows == 1 and n_cols == 1:
                axes = np.array([axes])
            elif n_rows == 1:
                axes = np.array(axes).reshape(1, -1)
            axes = axes.ravel()
            
            colors = sns.color_palette("coolwarm", n_colors=4)
            
            for idx, var in enumerate(var_set):
                suffix = '_pct_change' if use_percentage else '_change'
                var_change = var + suffix
                
                # Print variable header
                print(f"\n{self.var_labels[var]}:")
                
                sns.violinplot(
                    data=self.data,
                    x='STATUS_Label',
                    y=var_change,
                    hue='STATUS_Label',
                    inner='box',
                    ax=axes[idx],
                    palette=colors,
                    legend=False
                )
                
                axes[idx].set_title(f'Changes in {self.var_labels[var]}', pad=20)
                axes[idx].set_xlabel('Menopausal Status', labelpad=10)
                axes[idx].set_ylabel(
                    'Percent Change from Baseline' if use_percentage 
                    else 'Absolute Change from Baseline'
                )
                
                axes[idx].axhline(y=0, color='black', linestyle='--', alpha=0.3)
                axes[idx].tick_params(axis='x', rotation=45)
                
                # Calculate and print summary statistics
                summary_stats = self.data.groupby('STATUS_Label', observed=True)[var_change].agg([
                    'count', 'mean', 'std', 'median', 
                    ('min', 'min'), ('max', 'max')
                ]).round(3)
                
                # Print detailed statistics to terminal
                print("\nDetailed Statistics by Menopausal Stage:")
                print(summary_stats.to_string())
                
                # Calculate additional statistics
                for stage in summary_stats.index:
                    stage_data = self.data[self.data['STATUS_Label'] == stage][var_change]
                    
                    # Calculate confidence intervals
                    ci = stats.t.interval(
                        confidence=0.95, 
                        df=len(stage_data)-1,
                        loc=np.mean(stage_data),
                        scale=stats.sem(stage_data)
                    )
                    
                    print(f"\n{stage}:")
                    print(f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
                    
                    # Add statistical tests if you have more than 1 observation
                    if len(stage_data) > 1:
                        # One-sample t-test against 0 (testing if change is significant)
                        t_stat, p_val = stats.ttest_1samp(stage_data, 0)
                        print(f"One-sample t-test (H0: mean = 0):")
                        print(f"t-statistic: {t_stat:.3f}")
                        print(f"p-value: {p_val:.3f}")
                
                # Add summary statistics to plot
                stat_text = "Summary Statistics:\n"
                for status, stat_row in summary_stats.iterrows():
                    stat_text += f"\n{status}:\n"
                    stat_text += f"n={stat_row['count']}, mean={stat_row['mean']:.3f}\n"
                    stat_text += f"median={stat_row['median']:.3f}, sd={stat_row['std']:.3f}\n"
                
                axes[idx].text(
                    1.15, 0.5,
                    stat_text,
                    fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.8),
                    transform=axes[idx].transAxes,
                    verticalalignment='center'
                )
            
            # Remove empty subplots
            for j in range(idx + 1, len(axes)):
                axes[j].remove()
            
            fig.suptitle(
                f'{title} Changes Across Menopausal Stages',
                y=1.02,
                fontsize=14
            )
            
            plt.tight_layout()
            plt.savefig(
                f'{title.lower()}_changes_{"percent" if use_percentage else "absolute"}.png',
                dpi=300,
                bbox_inches='tight'
            )

            file_name = f'{self.output_dir}/{var}_violin_plot.png'
            plt.savefig(file_name)
            plt.close()
            
            print("\n" + "-"*80 + "\n")

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
            
            print("\nCalculating baseline changes...")
            self.calculate_baseline_changes()
            
            print("\nRunning GEE analysis...")
            self.run_gee_analysis()
            
            print("\nCreating distribution plots for absolute changes...")
            self.plot_change_distributions(use_percentage=False)
            
            print("\nCreating distribution plots for percentage changes...")
            self.plot_change_distributions(use_percentage=True)

            print("\nAnalysis complete")

        finally:
            sys.stdout = output_capture.terminal
            output_capture.close()

if __name__ == "__main__":
    file_path = "processed_combined_data.csv"
    analysis = MenopauseCognitionAnalysis(file_path)
    analysis.run_complete_analysis()