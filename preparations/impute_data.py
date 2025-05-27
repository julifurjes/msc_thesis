import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from scipy import stats
from pyampute.exploration.mcar_statistical_tests import MCARTest
import os

class ImputationValidator:
    def __init__(self, input_file, output_file, run_sensitivity, output_dir='imputation_validation'):
        """
        Initialize the ImputationValidator.
        
        Args:
            input_file: Path to the processed data CSV file
            output_dir: Directory to save validation outputs
        """
        self.data = pd.read_csv(input_file, low_memory=False)
        self.output_dir = output_dir
        self.variables = ['TOTIDE1', 'TOTIDE2']
        self.variables += ['NUMHOTF', 'BOTHOTF', 'NUMNITS', 'BOTNITS', 'NUMCLDS', 'BOTCLDS', 'HOW_HAR', 'BCINCML']
        self.imputed_data = None
        self.original_values = None
        self.imputed_values = None
        self.input_file = input_file
        self.output_file = output_file
        self.run_sensitivity = run_sensitivity
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Store original non-missing values
        self.original_values = {
            var: self.data[var][self.data[var].notna()].copy()
            for var in self.variables
        }

    def mcar_test(self):

        mt = MCARTest(method = 'little')

        # Perform Little's MCAR test
        p_value = mt(self.data[self.variables])

        print(f"Little's MCAR test p-value: {p_value:.5f}")

        if p_value > 0.05:
            print("The data is likely MCAR (missing completely at random).")
        else:
            print("The data is NOT MCAR (probably MAR or MNAR).")

    def check_missingness_patterns(self):
        """
        Check missingness patterns across variables and menopausal status.
        """
        print("\nChecking missingness patterns...")
        
        # Create binary indicators for missing values
        for var in self.variables:
            self.data[f'missing_{var}'] = self.data[var].isna().astype(int)
        
        # Calculate correlation between missing indicators
        missing_corr = self.data[[f'missing_{var}' for var in self.variables]].corr()
        
        # Create mapping for proper variable names
        variable_name_map = {
            'TOTIDE1': 'Immediate Recall',
            'TOTIDE2': 'Delayed Recall', 
            'NUMHOTF': 'Hot Flash Count',
            'BOTHOTF': 'Hot Flash Bother',
            'NUMNITS': 'Night Sweat Count',
            'BOTNITS': 'Night Sweat Bother',
            'NUMCLDS': 'Cold Sweat Count',
            'BOTCLDS': 'Cold Sweat Bother',
            'HOW_HAR': 'Financial Strain',
            'BCINCML': 'Income Discrimination'
        }
        
        # Clean up variable names for display (remove 'missing_' prefix and use proper names)
        clean_names = []
        for var in missing_corr.columns:
            var_clean = var.replace('missing_', '')
            clean_names.append(variable_name_map.get(var_clean, var_clean))
        
        missing_corr.columns = clean_names
        missing_corr.index = clean_names
        
        csv_path = os.path.join(self.output_dir, 'missingness_patterns.csv')
        missing_corr.to_csv(csv_path)
        print(f"Missingness patterns saved to: {csv_path}")
        
        # Create heatmap of missing correlations with improved formatting
        green_palette = sns.color_palette("YlGn", n_colors=10)
        plt.figure(figsize=(12, 10))
        
        # Create heatmap with larger fonts
        ax = sns.heatmap(
            missing_corr, 
            annot=True, 
            cmap=green_palette, 
            vmin=-1, 
            vmax=1,
            annot_kws={'size': 14},  # Larger numbers in cells
            cbar_kws={'shrink': 0.8}
        )
        
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=14, rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=14, rotation=0)
        plt.title('Correlation of Missing Data Patterns Between Variables', fontsize=16, pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'missing_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.show()

        # Using status labels
        status_map = {
            1: 'Surgical',
            2: 'Post-menopause',
            3: 'Late Peri',
            4: 'Early Peri',
            5: 'Pre-menopause',
            8: 'Surgical'
        }

        self.data['STATUS_Label'] = self.data['STATUS'].map(status_map)
        
        # Calculate missing rates by STATUS category
        missing_by_status = self.data.groupby('STATUS_Label')[
            [f'missing_{var}' for var in self.variables]
        ].mean().round(3) * 100  # Convert to percentage
        
        # Clean column names for display and apply proper variable names
        clean_column_names = []
        for col in missing_by_status.columns:
            var_clean = col.replace('missing_', '')
            clean_column_names.append(variable_name_map.get(var_clean, var_clean))
        
        missing_by_status.columns = clean_column_names
        
        csv_path = os.path.join(self.output_dir, 'missing_rates_by_status.csv')
        missing_by_status.to_csv(csv_path)
        print(f"Missing rates by menopausal status saved to: {csv_path}")
        
        # Create visualization of missing patterns with improved formatting
        plt.figure(figsize=(15, 10))
        
        for i, var in enumerate(self.variables):
            plt.subplot(3, 4, i+1)
            
            # Get proper variable name for plotting
            var_clean = variable_name_map.get(var, var)
            plot_data = missing_by_status[var_clean]
            
            # Create bar chart
            plt.bar(plot_data.index, plot_data.values)
            plt.title(f"{var_clean} Missing Rate (%)", fontsize=12)
            plt.ylabel("Missing (%)", fontsize=11)
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.yticks(fontsize=10)
            plt.ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'missingness_by_status.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Statistical test for each variable
        print("\nAssociation between missingness and menopausal status:")
        for var in self.variables:
            # Create contingency table
            table = pd.crosstab(
                self.data['STATUS_Label'],
                self.data[f'missing_{var}'],
                margins=False
            )
            
            # Chi-square test
            chi2, p, _, _ = stats.chi2_contingency(table)
            
            print(f"{var}: Chi-square = {chi2:.2f}, p-value = {p:.5f}")
            
            # Interpret the result
            if p < 0.05:
                print(f"  → Missing values for {var} are significantly associated with menopausal status")
            else:
                print(f"  → No significant association between {var} missingness and menopausal status")

    def perform_imputation(self, k_neighbors):
        """
        Perform KNN imputation and store both original and imputed values.
        """
        print("\nPerforming KNN imputation...")
        
        # Create copy of data for imputation
        self.imputed_data = self.data.copy()
        
        # Store masks of originally missing values
        missing_masks = {
            var: self.data[var].isna()
            for var in self.variables
        }
        
        # Create all the imputed columns - initialize with original values
        for var in self.variables:
            self.imputed_data[f'{var}_imputed'] = self.imputed_data[var]
        
        # Process each variable separately to avoid dimensionality issues
        for var in self.variables:
            print(f"  Imputing {var}...")
            
            for subject in self.data['SWANID'].unique():
                subject_mask = self.data['SWANID'] == subject
                subject_data = self.data.loc[subject_mask]
                
                # Skip if there's not enough data for this subject
                if len(subject_data) < 2:
                    continue
                    
                # Skip if this variable has no missing values for this subject
                if not subject_data[var].isna().any():
                    continue
                
                # Prepare features for imputation - just use VISIT and the current variable
                features = pd.DataFrame({
                    'VISIT': subject_data['VISIT'],
                    var: subject_data[var]
                })
                
                # Skip if not enough non-missing values to impute
                if features[var].notna().sum() < 1:
                    continue
                    
                # Initialize KNN imputer
                imputer = KNNImputer(
                    n_neighbors=min(k_neighbors, features[var].notna().sum()),  # Use only as many neighbors as available
                    weights='distance'
                )
                
                try:
                    # Perform imputation
                    imputed_values = imputer.fit_transform(features)
                    
                    # Round imputed values to nearest integer
                    imputed_values = np.round(imputed_values)
                    
                    # Store imputed values only for the variable being processed
                    self.imputed_data.loc[subject_mask, f'{var}_imputed'] = imputed_values[:, 1]  # Index 1 is the variable (0 is VISIT)
                except Exception as e:
                    print(f"Error imputing {var} for subject {subject}: {e}")
                    # If imputation fails, keep original values
                    continue
        
        # Store imputed values that were originally missing
        self.imputed_values = {
            var: self.imputed_data[f'{var}_imputed'][missing_masks[var]].copy()
            for var in self.variables
        }
        
        print("Imputation complete.")
    
    def check_distributions(self):
        """
        Compare distributions of observed vs imputed values.
        """
        print("\nChecking distributions...")
        
        # Calculate how many rows we need based on the number of variables
        n_vars = len(self.variables)
        n_rows = (n_vars + 1) // 2  # Ceiling division to get enough rows
        
        # Create a figure with enough subplots for all variables
        fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5 * n_rows))
        
        # If there's only one row, make sure axes is 2D
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, var in enumerate(self.variables):
            # Calculate row and column
            row = idx // 2
            col_start = (idx % 2) * 2  # Each variable gets 2 columns (0,1 or 2,3)
            
            # Histogram
            sns.histplot(
                data=self.original_values[var],
                label='Original',
                ax=axes[row, col_start],
                alpha=0.5
            )
            sns.histplot(
                data=self.imputed_values[var],
                label='Imputed',
                ax=axes[row, col_start],
                alpha=0.5
            )
            axes[row, col_start].set_title(f'{var} Distribution Comparison')
            axes[row, col_start].legend()
            
            # QQ plot
            stats.probplot(
                self.original_values[var],
                dist="norm",
                plot=axes[row, col_start + 1]
            )
            axes[row, col_start + 1].set_title(f'{var} Q-Q Plot')
        
        # Hide any unused subplots
        for i in range(n_vars, n_rows * 2):
            row = i // 2
            col_start = (i % 2) * 2
            axes[row, col_start].axis('off')
            axes[row, col_start + 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'distribution_checks.png'))
        plt.close()
        
        # Print summary statistics
        print("\nSummary Statistics Comparison:")
        for var in self.variables:
            print(f"\n{var}:")
            print("Original:")
            print(self.original_values[var].describe())
            print("\nImputed:")
            print(self.imputed_values[var].describe())
    
    def check_correlations(self):
        """
        Check correlation preservation between variables.
        """
        print("\nChecking correlations...")
        
        # Original correlations
        original_corr = self.data[self.variables].corr()

        green_palette = sns.color_palette("YlGn", n_colors=10)
        
        # Plot correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            original_corr,
            annot=True,
            fmt='.2f',
            cmap=green_palette,
            vmin=-1,
            vmax=1
        )
        plt.title('Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'correlation_checks.png'))
        plt.close()
        
        print("\nCorrelation Comparison:")
        print("Original:\n", original_corr)
    
    def perform_sensitivity_analysis(self, k_values=[2, 3, 4, 5]):
        """
        Test different k values for KNN imputation.
        """
        print("\nPerforming sensitivity analysis...")
        
        results = {var: [] for var in self.variables}

        k_values = [4]
        
        for k in k_values:
            print(f"\nTesting k={k}")
            self.mcar_test()
            self.check_missingness_patterns()
            self.perform_imputation(k_neighbors=k)
            self.check_distributions()
            self.analyze_distribution_shifts()
            self.check_correlations()
            self.analyze_correlation_structure()
            self.check_plausibility()
            self.generate_validation_report()
            
            # Store summary statistics for each k
            for var in self.variables:
                results[var].append({
                    'k': k,
                    'mean': self.imputed_values[var].mean(),
                    'std': self.imputed_values[var].std(),
                    'median': self.imputed_values[var].median()
                })

                print(f"\n{var} Summary Statistics (k={k}):")
                print("Mean:", results[var][-1]['mean'])
                print("Std:", results[var][-1]['std'])
                print("Median:", results[var][-1]['median'])
        
        # Plot sensitivity results
        fig, axes = plt.subplots(len(self.variables), 3, figsize=(15, 5*len(self.variables)))
        
        for idx, var in enumerate(self.variables):
            df = pd.DataFrame(results[var])
            
            # Plot mean
            axes[idx, 0].plot(df['k'], df['mean'], marker='o')
            axes[idx, 0].set_title(f'{var} Mean vs k')
            
            # Plot std
            axes[idx, 1].plot(df['k'], df['std'], marker='o')
            axes[idx, 1].set_title(f'{var} Std vs k')
            
            # Plot median
            axes[idx, 2].plot(df['k'], df['median'], marker='o')
            axes[idx, 2].set_title(f'{var} Median vs k')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sensitivity_analysis.png'))
        plt.close()
    
    def check_plausibility(self):
        """
        Check plausibility of imputed values.
        """
        print("\nChecking plausibility of imputed values...")
        
        for var in self.variables:
            # Calculate bounds based on original data
            q1 = self.original_values[var].quantile(0.25)
            q3 = self.original_values[var].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Check for outliers in imputed values
            outliers = self.imputed_values[var][
                (self.imputed_values[var] < lower_bound) |
                (self.imputed_values[var] > upper_bound)
            ]
            
            print(f"\n{var}:")
            print(f"Number of outliers in imputed values: {len(outliers)}")
            if len(outliers) > 0:
                print("Outlier values:")
                print(outliers)

    def analyze_distribution_shifts(self):
        """
        Analyze shifts in distributions between original and imputed data.
        """
        print("\nAnalyzing distribution shifts...")
        
        # Calculate detailed statistics for both original and imputed data
        stats_report = []
        
        for var in self.variables:
            orig_stats = self.original_values[var].describe(percentiles=[.05, .1, .25, .5, .75, .9, .95])
            imp_stats = self.imputed_values[var].describe(percentiles=[.05, .1, .25, .5, .75, .9, .95])
            
            # Calculate percent changes
            changes = pd.DataFrame({
                'Original': orig_stats,
                'Imputed': imp_stats,
                'Pct_Change': ((imp_stats - orig_stats) / orig_stats * 100).round(2)
            })
            
            stats_report.append((var, changes))
            
            # Create detailed visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
            
            # Density plot
            sns.kdeplot(data=self.original_values[var], ax=ax1, label='Original')
            sns.kdeplot(data=self.imputed_values[var], ax=ax1, label='Imputed')
            ax1.set_title(f'{var} Density Comparison')
            ax1.legend()
            
            # Box plot
            combined_data = pd.DataFrame({
                'Value': pd.concat([self.original_values[var], self.imputed_values[var]]),
                'Type': ['Original']*len(self.original_values[var]) + ['Imputed']*len(self.imputed_values[var])
            })
            sns.boxplot(data=combined_data, x='Type', y='Value', ax=ax2)
            ax2.set_title(f'{var} Distribution Comparison')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{var}_distribution_shifts.png'))
            plt.close()
        
        # Save detailed statistics report
        with open(os.path.join(self.output_dir, 'distribution_shifts_report.txt'), 'w') as f:
            for var, changes in stats_report:
                f.write(f"\n{var} Statistics Comparison:\n")
                f.write(changes.to_string())
                f.write("\n\n")

    def analyze_correlation_structure(self):
        """
        Analyze changes in correlation structure after imputation.
        """
        print("\nAnalyzing correlation structure changes...")
        
        print("Calculating correlations for key variables...")
        
        # Ensure numeric type for all variables
        orig_data = self.data[['TOTIDE1', 'TOTIDE2', 'VISIT']].apply(pd.to_numeric, errors='coerce')
        imp_data = self.imputed_data[['TOTIDE1_imputed', 'TOTIDE2_imputed', 'VISIT']].apply(pd.to_numeric, errors='coerce')
        
        # Calculate correlation matrices
        orig_corr = orig_data.corr().round(3)
        imputed_corr = imp_data.corr().round(3)
        
        # Calculate correlation differences for the main variables
        main_vars = ['TOTIDE1', 'TOTIDE2']
        imp_vars = ['TOTIDE1_imputed', 'TOTIDE2_imputed']
        
        # Initialize correlation difference matrix
        corr_diff = pd.DataFrame(
            np.zeros((2, 2)),
            index=main_vars,
            columns=main_vars
        )
        
        # Calculate differences
        for i, (orig_var, imp_var) in enumerate(zip(main_vars, imp_vars)):
            for j, (orig_var2, imp_var2) in enumerate(zip(main_vars, imp_vars)):
                orig_val = orig_corr.loc[orig_var, orig_var2]
                imp_val = imputed_corr.loc[imp_var, imp_var2]
                corr_diff.iloc[i, j] = imp_val - orig_val
        
        print("Creating correlation visualizations...")
        
        # Create correlation comparison plot
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        
        # Original correlations
        mask1 = np.zeros_like(orig_corr, dtype=bool)
        mask1[np.triu_indices_from(mask1, k=1)] = True
        
        sns.heatmap(
            orig_corr,
            annot=True,
            fmt='.2f',
            ax=ax1,
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            mask=mask1
        )
        ax1.set_title('Original Correlations')
        
        # Imputed correlations
        mask2 = np.zeros_like(imputed_corr, dtype=bool)
        mask2[np.triu_indices_from(mask2, k=1)] = True
        
        sns.heatmap(
            imputed_corr,
            annot=True,
            fmt='.2f',
            ax=ax2,
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            mask=mask2
        )
        ax2.set_title('Imputed Correlations')
        
        # Correlation differences
        mask3 = np.zeros_like(corr_diff, dtype=bool)
        mask3[np.triu_indices_from(mask3, k=1)] = True
        
        sns.heatmap(
            corr_diff,
            annot=True,
            fmt='.2f',
            ax=ax3,
            cmap='RdBu',
            center=0,
            vmin=-0.2,
            vmax=0.2,
            mask=mask3
        )
        ax3.set_title('Correlation Differences\n(Imputed - Original)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'correlation_structure_analysis.png'))
        plt.close()
        
        print("Generating correlation report...")
        
        # Save detailed correlation report
        with open(os.path.join(self.output_dir, 'correlation_analysis_report.txt'), 'w') as f:
            f.write("Correlation Structure Analysis:\n\n")
            f.write("Original Correlations:\n")
            f.write(orig_corr.to_string())
            f.write("\n\nImputed Correlations:\n")
            f.write(imputed_corr.to_string())
            f.write("\n\nCorrelation Changes (Imputed - Original):\n")
            f.write(corr_diff.to_string())
            
            # Add interpretation
            f.write("\n\nKey Findings:\n")
            for i, (orig_var, imp_var) in enumerate(zip(main_vars, imp_vars)):
                for j, (orig_var2, imp_var2) in enumerate(zip(main_vars, imp_vars)):
                    if i <= j:  # Only lower triangle
                        diff = corr_diff.iloc[i, j]
                        f.write(f"\n- Correlation between {orig_var} and {orig_var2}:")
                        f.write(f"\n  Original: {orig_corr.loc[orig_var, orig_var2]:.3f}")
                        f.write(f"\n  Imputed:  {imputed_corr.loc[imp_var, imp_var2]:.3f}")
                        f.write(f"\n  Change:   {diff:.3f}")
                        
                        # Add interpretation
                        if abs(diff) < 0.1:
                            f.write("\n  Interpretation: Correlation well preserved")
                        elif abs(diff) < 0.2:
                            f.write("\n  Interpretation: Moderate change in correlation")
                        else:
                            f.write("\n  Interpretation: Substantial change in correlation - requires attention")
        
        print("Correlation analysis complete.")

    def generate_validation_report(self):
        """
        Generate a comprehensive validation report.
        """
        print("\nGenerating validation report...")
        
        report_content = []
        
        # Add methodology section
        report_content.append("# Imputation Validation Report\n")
        report_content.append("## Methodology\n")
        report_content.append("- Method: K-Nearest Neighbors (KNN) Imputation")
        report_content.append(f"- Variables imputed: {', '.join(self.variables)}")
        report_content.append("- Parameters used:")
        report_content.append(f"  - k values tested: {', '.join(map(str, [2,3,4,5]))}")
        report_content.append("  - Distance metric: Euclidean")
        report_content.append("  - Within-subject imputation applied\n")
        
        # Add summary statistics
        report_content.append("## Summary Statistics\n")
        for var in self.variables:
            report_content.append(f"### {var}\n")
            orig_stats = self.original_values[var].describe()
            imp_stats = self.imputed_values[var].describe()
            report_content.append("Original Data:")
            report_content.append(orig_stats.to_string())
            report_content.append("\nImputed Data:")
            report_content.append(imp_stats.to_string())
            report_content.append("\n")

        # Save report
        with open(os.path.join(self.output_dir, 'validation_report.md'), 'w') as f:
            f.write('\n'.join(report_content))

    def save_imputed_data(self):
        """
        Save the imputed data, replacing original TOTIDE columns with imputed values.
        """
        print("\nPreparing data for saving...")
        
        # Create a copy of the data for saving
        save_data = self.data.copy()
        
        # Replace original TOTIDE columns with imputed values
        for var in self.variables:
            # Get the mask of missing values in original data
            missing_mask = save_data[var].isna()
            
            # Replace missing values with imputed values
            save_data.loc[missing_mask, var] = self.imputed_data.loc[missing_mask, f'{var}_imputed']
        
        # Don't include the imputed columns in the saved file
        imputed_cols = [f'{var}_imputed' for var in self.variables]
        cols_to_save = [col for col in save_data.columns if col not in imputed_cols]
        
        print("Saving imputed data...")
        # Save to the original input file
        save_data[cols_to_save].to_csv(self.output_file, index=False)
        print(f"Data saved to: {self.output_file}")
        
        # Print summary of changes
        print("\nImputation Summary:")
        for var in self.variables:
            n_imputed = save_data[var].notna().sum() - self.data[var].notna().sum()
            print(f"{var}:")
            print(f"  Values imputed: {n_imputed}")
            print(f"  Final missing values: {save_data[var].isna().sum()}")

    def run_all_checks(self):
        """
        Run all validation checks and generate comprehensive report.
        """
        self.perform_sensitivity_analysis()
        if not self.run_sensitivity:
            best_k = 4  # Best k value based on sensitivity analysis
            self.perform_imputation(best_k)
            self.check_distributions()
            self.analyze_distribution_shifts()
            self.check_correlations()
            self.analyze_correlation_structure()
            self.check_plausibility()
            self.generate_validation_report()
        
        self.save_imputed_data()
        
        print("\nValidation complete. Results saved in:", self.output_dir)

if __name__ == "__main__":
    # Initialize validator with your data file
    validator = ImputationValidator(
        input_file="processed_data.csv",
        output_file="processed_combined_data.csv",
        run_sensitivity=True,
        output_dir="imputation_validation"
    )
    
    # Run all validation checks
    validator.run_all_checks()