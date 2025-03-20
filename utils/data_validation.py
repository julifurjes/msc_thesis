from typing import List, Dict, Optional, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import normaltest, shapiro, levene
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.save_output import OutputCapture, get_output_dir

class DataValidator:
    """Universal data validation class for statistical analysis."""
    
    def __init__(self, data: pd.DataFrame, 
                 variables: List[str] = None,
                 output_dir: str = None,
                 plotting: bool = False):
        """
        Initialize DataValidator.
        
        Args:
            data (pd.DataFrame): Data to validate
            variables (List[str], optional): Variables to validate. If None, uses all columns
            output_dir (str, optional): Directory to save plots
            output_capture (OutputCapture, optional): OutputCapture instance for text output
        """
        self.data = data
        self.variables = variables if variables is not None else data.columns.tolist()
        self.base_output_dir = output_dir if output_dir else '.'
        self.output_dir = os.path.join(self.base_output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.plotting = plotting
        self.validation_results = {}
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def save_plot(self, plot_name: str):
        """Save plot to output directory if save_plots is True."""
        plt.savefig(os.path.join(self.output_dir, plot_name))
        plt.close()
    
    def check_missing_values(self) -> Dict:
        """Check for missing values in specified variables."""
        missing_data = self.data[self.variables].isnull().sum()
        missing_percentages = (missing_data / len(self.data)) * 100
        
        return {
            'counts': missing_data.to_dict(),
            'percentages': missing_percentages.to_dict()
        }
    
    def check_distributions(self) -> Dict:
        """
        Check distributions of variables.
        """
        results = {}
        
        for var in self.variables:
            # Clean the data: remove whitespace and empty strings
            if self.data[var].dtype == 'object':
                cleaned_data = self.data[var].astype(str).str.strip()
                cleaned_data = cleaned_data[cleaned_data != '']
            else:
                cleaned_data = self.data[var]
            
            # Drop NA values after cleaning
            cleaned_data = cleaned_data.dropna()
            
            if len(cleaned_data) == 0:
                print(f"Warning: No valid data for {var} after cleaning")
                continue
                
            try:
                # Try numeric analysis
                numeric_data = pd.to_numeric(cleaned_data, errors='raise')
                stats_dict = {
                    'type': 'numeric',
                    'basic_stats': {
                        'mean': numeric_data.mean(),
                        'median': numeric_data.median(),
                        'std': numeric_data.std(),
                        'skewness': stats.skew(numeric_data),
                        'kurtosis': stats.kurtosis(numeric_data)
                    }
                }
                
                # Normality tests
                if len(numeric_data) >= 3:  # Minimum required for normality tests
                    try:
                        _, shapiro_p = shapiro(numeric_data[:5000])
                        _, normal_p = normaltest(numeric_data)
                        stats_dict['normality'] = {
                            'shapiro_p': shapiro_p,
                            'dagostino_p': normal_p
                        }
                    except Exception as e:
                        print(f"Warning: Could not perform normality tests for {var}: {str(e)}")
                        stats_dict['normality'] = None
                
                if self.plotting:
                    self._plot_numeric_distribution(numeric_data, var)
                    
            except (ValueError, TypeError) as e:
                # Categorical analysis
                value_counts = cleaned_data.value_counts()
                total = len(cleaned_data)
                proportions = value_counts / total if total > 0 else value_counts * 0
                
                stats_dict = {
                    'type': 'categorical',
                    'counts': value_counts.to_dict(),
                    'proportions': proportions.to_dict()
                }
                
                if self.plotting:
                    self._plot_categorical_distribution(value_counts, var)
            
            except Exception as e:
                print(f"Warning: Error analyzing distribution for {var}: {str(e)}")
                continue
            
            results[var] = stats_dict
            
        return results
    
    def check_group_sizes(self, grouping_var: str) -> Dict:
        """Check sample sizes within groups."""
        if grouping_var not in self.data.columns:
            return None
            
        results = {}
        for var in self.variables:
            group_counts = self.data.groupby(grouping_var)[var].count()
            min_size = group_counts.min()
            max_size = group_counts.max()
            ratio = max_size / min_size if min_size > 0 else float('inf')
            
            results[var] = {
                'counts': group_counts.to_dict(),
                'min_size': min_size,
                'max_size': max_size,
                'ratio': ratio,
                'balanced': ratio < 1.5
            }
            
        return results
        
    def check_homogeneity(self, grouping_var: str) -> Dict:
        """
        Check homogeneity of variance between groups using Levene's test.
        """
        if grouping_var not in self.data.columns:
            print(f"Warning: {grouping_var} not found in data")
            return None
            
        levene_results = {}
        
        for var in self.variables:
            try:
                # Get groups and ensure they're numeric
                self.data[var] = pd.to_numeric(self.data[var], errors='coerce')

                groups = []
                group_names = []
                group_data = []
                
                for name, group in self.data.groupby(grouping_var):
                    # Convert to numeric and drop NA
                    values = pd.to_numeric(group[var], errors='coerce').dropna()
                    if len(values) > 1:  # Need at least 2 points per group
                        groups.append(values.values)
                        # Convert name to string to avoid type issues with plot labels
                        str_name = str(name)
                        group_names.append(str_name)
                        group_data.append({
                            'group': str_name,  # Store as string
                            'values': values,
                            'mean': values.mean(),
                            'residuals': values - values.mean()
                        })
                
                if len(groups) >= 2:  # Need at least 2 groups
                    # Convert all groups to numpy arrays
                    groups = [np.array(g, dtype=float) for g in groups]
                    stat, p_value = levene(*groups)

                    # Create residual plots
                    if self.plotting:
                        plot_filename = self._create_residual_plots(var, group_data, grouping_var)
                        print(f"Residual plots saved as: {plot_filename}")
                    
                    levene_results[var] = {
                        'statistic': float(stat),
                        'p_value': float(p_value),
                        'homogeneous': p_value > 0.05,
                        'groups_tested': group_names
                    }
                    
                    print(f"\nLevene's test for {var}:")
                    print(f"Statistic: {stat:.4f}")
                    print(f"p-value: {p_value:.4f}")
                    print(f"Groups tested: {', '.join(group_names)}")
                else:
                    print(f"\nWarning: Not enough valid groups for Levene's test on {var}")
                    levene_results[var] = None
                    
            except Exception as e:
                print(f"\nWarning: Could not perform Levene's test for {var}: {str(e)}")
                levene_results[var] = None
        
        return levene_results
    
    def _create_residual_plots(self, var: str, group_data: List[Dict], grouping_var: str) -> str:
        """
        Create comprehensive residual plots for homogeneity analysis.
        
        Parameters:
        -----------
        var : str
            Variable name
        group_data : List[Dict]
            List of dictionaries containing group data
        grouping_var : str
            Name of the grouping variable
            
        Returns:
        --------
        str
            Filename of saved plot
        """
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(2, 2)
        
        # 1. Residuals vs. Groups (Box Plot)
        ax1 = fig.add_subplot(gs[0, 0])
        box_data = [d['residuals'] for d in group_data]
        ax1.boxplot(box_data, labels=[d['group'] for d in group_data])
        ax1.set_title('Residuals by Group')
        ax1.set_xlabel(grouping_var)
        ax1.set_ylabel('Residuals')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Residuals vs. Fitted Values (Scatter Plot)
        ax2 = fig.add_subplot(gs[0, 1])
        for d in group_data:
            ax2.scatter([d['mean']] * len(d['residuals']), d['residuals'], 
                    alpha=0.5, label=d['group'])
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        ax2.set_title('Residuals vs. Fitted Values')
        ax2.set_xlabel('Fitted Values')
        ax2.set_ylabel('Residuals')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 3. Q-Q Plot of Residuals
        ax3 = fig.add_subplot(gs[1, 0])
        all_residuals = np.concatenate([d['residuals'] for d in group_data])
        stats.probplot(all_residuals, plot=ax3)
        ax3.set_title('Q-Q Plot of Residuals')
        
        # 4. Residual Density Plot by Group
        ax4 = fig.add_subplot(gs[1, 1])
        for d in group_data:
            sns.kdeplot(data=d['residuals'], ax=ax4, label=d['group'])
        ax4.set_title('Residual Density by Group')
        ax4.set_xlabel('Residuals')
        ax4.set_ylabel('Density')
        
        # Add overall title
        plt.suptitle(f'Residual Analysis for {var}', y=1.02, fontsize=14)
        
        # Add homogeneity test results
        stat, p_value = levene(*[d['residuals'] for d in group_data])
        fig.text(0.02, 0.02, 
                f"Levene's test:\nstatistic = {stat:.4f}\np-value = {p_value:.4f}",
                bbox=dict(facecolor='white', alpha=0.8))
        
        # Save plot
        filename = os.path.join(self.output_dir, f'{var}_residual_plots.png')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def check_multicollinearity(self) -> Optional[Dict]:
        """
        Check for multicollinearity between numeric variables using correlation matrix and VIF.
            
        Returns:
            Dict: Dictionary containing correlation matrix and VIF scores.
        """
        # Convert to numeric and drop non-numeric
        numeric_data = self.data[self.variables].apply(pd.to_numeric, errors='coerce')
        
        if numeric_data.shape[1] < 2:
            print("Warning: Not enough numeric variables for multicollinearity check.")
            return None
        
        # Calculate correlation matrix
        corr_matrix = numeric_data.corr()

        # Drop any rows with NaN values
        clean_numeric_data = numeric_data.dropna()
        
        # Check if we have enough data after dropping NaNs
        if clean_numeric_data.shape[0] <= 1:
            print("Not enough complete rows for VIF calculation after removing missing values")
            return {
                'correlation_matrix': corr_matrix.to_dict(),
                'vif_scores': "Unable to calculate VIF scores due to missing data",
                'high_vif_warning': []
            }
        
        # Check for infinite values and replace them
        clean_numeric_data = clean_numeric_data.replace([np.inf, -np.inf], np.nan)
        clean_numeric_data = clean_numeric_data.dropna()
        
        if clean_numeric_data.shape[0] <= 1:
            print("Not enough complete rows for VIF calculation after removing infinite values")
            return {
                'correlation_matrix': corr_matrix.to_dict(),
                'vif_scores': "Unable to calculate VIF scores due to infinite values",
                'high_vif_warning': []
            }
        
        # Calculate VIF for each variable
        vif_data = pd.DataFrame()
        vif_data["Variable"] = clean_numeric_data.columns
        vif_data["VIF"] = [variance_inflation_factor(clean_numeric_data.values, i) 
                        for i in range(numeric_data.shape[1])]
        
        # Sort VIF scores in descending order
        vif_data = vif_data.sort_values('VIF', ascending=False)
        
        if self.plotting:
            # Plot correlation matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Matrix')
            self.save_plot('correlation_matrix.png')
            
            # Plot VIF scores
            plt.figure(figsize=(10, 6))
            sns.barplot(data=vif_data, x='Variable', y='VIF')
            plt.xticks(rotation=45)
            plt.title('Variance Inflation Factors')
            plt.tight_layout()
            self.save_plot('vif_scores.png')
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'vif_scores': vif_data.to_dict('records'),
            'high_vif_warning': vif_data[vif_data['VIF'] > 5]['Variable'].tolist()
        }
    
    def check_independence(self, grouping_var: str = None) -> Dict:
        """
        Check independence of observations using lag-1 correlation.
        
        Args:
            grouping_var (str, optional): If provided, checks independence within each group
        """
        independence_results = {}
        
        for var in self.variables:
            try:
                if grouping_var and grouping_var in self.data.columns:
                    # Check independence within each group
                    group_results = {}
                    for name, group in self.data.groupby(grouping_var):
                        data = pd.to_numeric(group[var], errors='coerce').dropna()
                        if len(data) >= 3:  # Need at least 3 points for lag correlation
                            # Calculate lag-1 correlation
                            lag_corr = np.corrcoef(data[:-1], data[1:])[0, 1]
                            # Consider independent if correlation is weak (|r| < 0.3)
                            independent = abs(lag_corr) < 0.3
                            
                            group_results[str(name)] = {
                                'lag_correlation': float(lag_corr),
                                'independent': independent,
                                'n_observations': len(data)
                            }
                            
                            print(f"\nIndependence check for {var} in group {name}:")
                            print(f"Lag-1 correlation: {lag_corr:.4f}")
                            print(f"N observations: {len(data)}")
                        else:
                            print(f"\nWarning: Not enough observations for independence check in group {name}")
                    
                    independence_results[var] = group_results if group_results else None
                else:
                    # Check independence for whole variable
                    data = pd.to_numeric(self.data[var], errors='coerce').dropna()
                    if len(data) >= 3:
                        # Calculate lag-1 correlation
                        lag_corr = np.corrcoef(data[:-1], data[1:])[0, 1]
                        # Consider independent if correlation is weak (|r| < 0.3)
                        independent = abs(lag_corr) < 0.3
                        
                        independence_results[var] = {
                            'lag_correlation': float(lag_corr),
                            'independent': independent,
                            'n_observations': len(data)
                        }
                        
                        print(f"\nIndependence check for {var}:")
                        print(f"Lag-1 correlation: {lag_corr:.4f}")
                        print(f"N observations: {len(data)}")
                    else:
                        print(f"\nWarning: Not enough observations for independence check of {var}")
                        independence_results[var] = None
                    
            except Exception as e:
                print(f"\nWarning: Could not perform independence check for {var}: {str(e)}")
                independence_results[var] = None
        
        return independence_results
    
    def _plot_numeric_distribution(self, data: pd.Series, var_name: str):
        """Create distribution plots for numeric variables."""
        plt.figure(figsize=(12, 6))
        
        # Histogram with KDE
        plt.subplot(1, 2, 1)
        sns.histplot(data=data, kde=True)
        plt.title(f'Distribution of {var_name}')
        
        # Q-Q plot
        plt.subplot(1, 2, 2)
        stats.probplot(data, dist="norm", plot=plt)
        plt.title(f'Q-Q Plot of {var_name}')
        
        plt.tight_layout()
        self.save_plot(f'distribution_{var_name}.png')
    
    def _plot_categorical_distribution(self, value_counts: pd.Series, var_name: str):
        """Create distribution plots for categorical variables."""
        plt.figure(figsize=(10, 6))
        sns.barplot(x=value_counts.index, y=value_counts.values)
        plt.title(f'Distribution of {var_name}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        self.save_plot(f'distribution_{var_name}_categorical.png')

    def generate_validation_summary(self, results: Dict) -> str:
        """
        Generate a comprehensive validation summary as a string.
        
        Args:
            results (Dict): Results from validation checks
            
        Returns:
            str: Formatted summary text
        """
        summary = ["=== Data Validation Summary ===\n"]
        
        # Missing Values
        if 'missing' in results:
            summary.append("\n1. Missing Data Summary:")
            for var, percentage in results['missing']['percentages'].items():
                summary.append(f"{var}: {percentage:.2f}% missing")
        
        # Distributions
        if 'distributions' in results:
            summary.append("\n2. Distribution Summary:")
            for var, var_results in results['distributions'].items():
                if var_results:
                    summary.append(f"\n{var} ({var_results['type']}):")
                    if var_results['type'] == 'numeric':
                        stats = var_results['basic_stats']
                        summary.append(f"  Mean: {stats['mean']:.2f}")
                        summary.append(f"  Median: {stats['median']:.2f}")
                        summary.append(f"  Std: {stats['std']:.2f}")
                        summary.append(f"  Skewness: {stats['skewness']:.2f}")
                        summary.append(f"  Kurtosis: {stats['kurtosis']:.2f}")
                        if 'normality' in var_results and var_results['normality'] is not None:
                            summary.append(f"  Shapiro-Wilk p-value: {var_results['normality']['shapiro_p']:.2e}")
                            summary.append(f"  D'Agostino p-value: {var_results['normality']['dagostino_p']:.2e}")
                    else:  # categorical
                        summary.append(f"  Number of categories: {len(var_results['counts'])}")
                        # Show top 3 most frequent categories
                        sorted_cats = sorted(var_results['proportions'].items(), 
                                        key=lambda x: x[1], reverse=True)[:3]
                        for cat, prop in sorted_cats:
                            summary.append(f"  {cat}: {prop:.1%}")
        
        # Group Sizes
        if 'group_sizes' in results:
            summary.append("\n3. Group Size Analysis:")
            for var, size_results in results['group_sizes'].items():
                if size_results:  # Check if not None
                    summary.append(f"\n{var}:")
                    summary.append(f"  Min group size: {size_results['min_size']}")
                    summary.append(f"  Max group size: {size_results['max_size']}")
                    summary.append(f"  Max/Min ratio: {size_results['ratio']:.2f}")
                    summary.append(f"  Balance status: {'Balanced' if size_results['balanced'] else 'Imbalanced'}")
                    
                    summary.append("\n  Group counts:")
                    for group, count in size_results['counts'].items():
                        summary.append(f"    {group}: {count}")
        
        # Homogeneity
        if 'homogeneity' in results:
            summary.append("\n4. Homogeneity of Variance:")
            for var, var_results in results['homogeneity'].items():
                if var_results:  # Check if not None
                    summary.append(f"{var}: p-value = {var_results['p_value']:.4f} "
                                f"({'Homogeneous' if var_results['homogeneous'] else 'Non-homogeneous'})")
        
        # Independence
        if 'independence' in results:
            summary.append("\n5. Independence Check Results:")
            for var, var_results in results['independence'].items():
                if var_results:  # Check if not None
                    if isinstance(var_results, dict) and 'lag_correlation' in var_results:
                        summary.append(f"{var}: lag-1 correlation = {var_results['lag_correlation']:.4f} "
                                    f"({'Independent' if var_results['independent'] else 'Dependent'})")
                    elif isinstance(var_results, dict):  # Group results
                        summary.append(f"\n{var} (by group):")
                        for group, group_result in var_results.items():
                            if group_result and 'lag_correlation' in group_result:
                                summary.append(f"  {group}: lag-1 correlation = {group_result['lag_correlation']:.4f} "
                                            f"({'Independent' if group_result['independent'] else 'Dependent'})")
        
        # Multicollinearity
        if 'multicollinearity' in results and results['multicollinearity'] is not None:
            summary.append("\n6. Multicollinearity Analysis:")
            
            # Handle correlation matrix
            if 'correlation_matrix' in results['multicollinearity']:
                try:
                    # Create a DataFrame from the correlation matrix dictionary
                    corr_data = results['multicollinearity']['correlation_matrix']
                    
                    # Check if corr_data is a dictionary of dictionaries (proper matrix format)
                    if isinstance(corr_data, dict) and all(isinstance(v, dict) for v in corr_data.values()):
                        summary.append("  Correlation matrix analysis:")
                        
                        # Find high correlations
                        high_corr = []
                        variables = list(corr_data.keys())
                        
                        for i, var1 in enumerate(variables):
                            for j, var2 in enumerate(variables):
                                if i < j:  # Only check upper triangle to avoid duplicates
                                    # Safe access with default
                                    try:
                                        corr = corr_data[var1].get(var2, 0)
                                        if abs(corr) > 0.7:  # Common threshold for high correlation
                                            high_corr.append(f"  High correlation between {var1} and {var2}: {corr:.2f}")
                                    except Exception as e:
                                        # Skip any problematic pairs
                                        pass
                        
                        if high_corr:
                            summary.append("  High correlations detected:")
                            summary.extend(high_corr)
                        else:
                            summary.append("  No high correlations detected (threshold: 0.7)")
                    else:
                        summary.append("  Correlation matrix data format is not as expected")
                except Exception as e:
                    summary.append(f"  Error processing correlation matrix: {str(e)}")
            
            # Handle VIF scores
            if 'vif_scores' in results['multicollinearity']:
                vif_scores = results['multicollinearity']['vif_scores']
                if isinstance(vif_scores, list):
                    summary.append("\n  Variance Inflation Factors (VIF):")
                    try:
                        for item in vif_scores:
                            if isinstance(item, dict) and 'Variable' in item and 'VIF' in item:
                                summary.append(f"    {item['Variable']}: {item['VIF']}")
                    except Exception as e:
                        summary.append(f"  Error processing VIF scores: {str(e)}")
                elif isinstance(vif_scores, str):
                    # If vif_scores is a string message
                    summary.append(f"\n  VIF analysis: {vif_scores}")
            
            # Handle high VIF warning
            if 'high_vif_warning' in results['multicollinearity']:
                high_vif = results['multicollinearity']['high_vif_warning']
                if high_vif and isinstance(high_vif, list):
                    summary.append("\n  Variables with high VIF (>5):")
                    for var in high_vif:
                        summary.append(f"    {var}")
                elif isinstance(high_vif, list) and not high_vif:
                    summary.append("\n  No variables with high VIF detected")
        
        return '\n'.join(summary)
    
    def run_checks(self, checks: List[str], **kwargs) -> Dict:
        """
        Run specified validation checks.
        
        Args:
            checks (List[str]): List of checks to run
            **kwargs: Additional arguments for specific checks
        """
        output_capture = OutputCapture(self.output_dir, 'data_validation_results.txt')
        sys.stdout = output_capture

        try:
            available_checks = {
                'missing': self.check_missing_values,
                'distributions': self.check_distributions,
                'group_sizes': self.check_group_sizes,
                'homogeneity': self.check_homogeneity,
                'multicollinearity': self.check_multicollinearity,
                'independence': self.check_independence
            }

            print("=== Data Validation Results ===")
            
            results = {}

            for check in checks:
                print(f"DEBUG: Processing check: {check}")
                if check in available_checks:
                    print(f"DEBUG: {check} is a valid check")
                    if check in ['group_sizes', 'homogeneity']:
                        print(f"DEBUG: {check} needs grouping_var")
                        if 'grouping_var' not in kwargs:
                            print(f"DEBUG: grouping_var not found in kwargs")
                            print(f"Warning: {check} requires grouping_var parameter")
                            continue
                        results[check] = available_checks[check](kwargs['grouping_var'])
                        print(f"DEBUG: {check} completed successfully")
                    elif check == 'independence' and 'time_var' in kwargs:
                        results[check] = available_checks[check](kwargs['time_var'])
                        print(f"DEBUG: independence completed successfully")
                    else:
                        results[check] = available_checks[check]()
                        print(f"DEBUG: {check} completed successfully")
                else:
                    print(f"DEBUG: Invalid check: {check}, skipping")
            
            summary = self.generate_validation_summary(results)
            print(summary)
            
            return results
            
        except Exception as e:
            print(f"DEBUG ERROR: Exception in run_checks: {str(e)}")
            import traceback
            print(f"DEBUG ERROR: Traceback: {traceback.format_exc()}")
            return {}

        finally:
            sys.stdout = output_capture.terminal
            output_capture.close()
            return results