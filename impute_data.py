import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from scipy import stats
import os

class ImputationValidator:
    def __init__(self, input_file, output_file, output_dir='imputation_validation', only_run_sensitivity=False):
        """
        Initialize the ImputationValidator.
        
        Args:
            input_file: Path to the processed data CSV file
            output_dir: Directory to save validation outputs
        """
        self.data = pd.read_csv(input_file, low_memory=False)
        self.output_dir = output_dir
        self.variables = ['TOTIDE1', 'TOTIDE2']
        self.imputed_data = None
        self.original_values = None
        self.imputed_values = None
        self.input_file = input_file
        self.output_file = output_file
        self.only_run_sensitivity = only_run_sensitivity
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Store original non-missing values
        self.original_values = {
            var: self.data[var][self.data[var].notna()].copy()
            for var in self.variables
        }
    
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
        
        # Process each subject separately
        for subject in self.data['SWANID'].unique():
            subject_mask = self.data['SWANID'] == subject
            subject_data = self.data.loc[subject_mask]
            
            if len(subject_data) < 2:
                continue
            
            # Prepare features for imputation
            features = pd.DataFrame({
                'VISIT': subject_data['VISIT'],
                'TOTIDE1': subject_data['TOTIDE1'],
                'TOTIDE2': subject_data['TOTIDE2']
            })
            
            # Initialize KNN imputer
            imputer = KNNImputer(
                n_neighbors=min(k_neighbors, len(features)),
                weights='distance'
            )
            
            # Perform imputation
            imputed_values = imputer.fit_transform(features)

            # Round imputed values to nearest integer due to the nature of the data
            imputed_values = np.round(imputed_values)
            
            # Store imputed values in new columns
            for idx, var in enumerate(['TOTIDE1', 'TOTIDE2'], 1):
                self.imputed_data.loc[subject_mask, f'{var}_imputed'] = imputed_values[:, idx]
        
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
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for idx, var in enumerate(self.variables):
            # Histogram
            sns.histplot(
                data=self.original_values[var],
                label='Original',
                ax=axes[idx, 0],
                alpha=0.5
            )
            sns.histplot(
                data=self.imputed_values[var],
                label='Imputed',
                ax=axes[idx, 0],
                alpha=0.5
            )
            axes[idx, 0].set_title(f'{var} Distribution Comparison')
            axes[idx, 0].legend()
            
            # QQ plot
            stats.probplot(
                self.original_values[var],
                dist="norm",
                plot=axes[idx, 1]
            )
            axes[idx, 1].set_title(f'{var} Q-Q Plot')
        
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
        
        # Imputed correlations
        imputed_corr = self.imputed_data[[f'{var}_imputed' for var in self.variables]].corr()
        
        # Plot correlation matrices
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        sns.heatmap(original_corr, annot=True, ax=ax1)
        ax1.set_title('Original Correlations')
        
        sns.heatmap(imputed_corr, annot=True, ax=ax2)
        ax2.set_title('Imputed Correlations')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'correlation_checks.png'))
        plt.close()
        
        print("\nCorrelation Comparison:")
        print("Original:\n", original_corr)
        print("\nImputed:\n", imputed_corr)
    
    def perform_sensitivity_analysis(self, k_values=[2, 3, 4, 5]):
        """
        Test different k values for KNN imputation.
        """
        print("\nPerforming sensitivity analysis...")
        
        results = {var: [] for var in self.variables}
        
        for k in k_values:
            print(f"\nTesting k={k}")
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
        
        # Define relevant columns for correlation analysis
        relevant_cols = [
            'TOTIDE1', 'TOTIDE2',                      # Original variables
            'TOTIDE1_imputed', 'TOTIDE2_imputed',      # Imputed variables
            'VISIT'                                     # Important contextual variable
        ]
        
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
        if self.only_run_sensitivity:
            self.perform_sensitivity_analysis()
        else:
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
        output_dir="imputation_validation",
        only_run_sensitivity=False
    )
    
    # Run all validation checks
    validator.run_all_checks()