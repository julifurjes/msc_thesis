from typing import List, Dict, Optional, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import normaltest, shapiro
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
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
            variables (List[str], optional): Variables to validate. If None, uses all columns.
            output_dir (str, optional): Directory to save plots.
        """
        self.data = data
        self.variables = variables if variables is not None else data.columns.tolist()
        self.base_output_dir = output_dir if output_dir else '.'
        self.output_dir = os.path.join(self.base_output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.plotting = plotting
        self.validation_results = {}
    
    def save_plot(self, plot_name: str):
        """Save plot to output directory."""
        plt.savefig(os.path.join(self.output_dir, plot_name))
        plt.close()
    
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
    
    def check_group_sizes(self) -> Dict:
        """Check sample sizes for each value within each variable."""
        results = {}

        # Only check a subset of variables (demographic and grouping variables)
        variables_to_check = ['STATUS', 'INCOME', 'DEGREE']
        
        for var in variables_to_check:
            # Use each variable's values as groups
            var_values = self.data[var].dropna().unique()
            group_counts = self.data[var].value_counts().to_dict()
            
            min_size = min(group_counts.values()) if group_counts else 0
            max_size = max(group_counts.values()) if group_counts else 0
            ratio = max_size / min_size if min_size > 0 else float('inf')
            
            results[var] = {
                'counts': group_counts,
                'min_size': min_size,
                'max_size': max_size,
                'balanced': ratio < 1.5,
                'unique_values': len(var_values)
            }
            
        return results
        
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
            print("Not enough complete rows for VIF calculation after removing missing data")
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
    
    def check_stationarity(self) -> Dict:
        """
        Check stationarity of time series variables using ADF and KPSS tests.
        
        Returns:
            Dict: A dictionary with stationarity test results for each variable.
        """
        stationarity_results = {}
        
        for var in self.variables:
            data_series = pd.to_numeric(self.data[var], errors='coerce').dropna()
            if len(data_series) < 10:
                print(f"Warning: Not enough observations for stationarity test for {var}")
                continue
            
            var_results = {}
            try:
                adf_result = adfuller(data_series)
                var_results['adf_statistic'] = adf_result[0]
                var_results['adf_pvalue'] = adf_result[1]
            except Exception as e:
                print(f"Warning: ADF test failed for {var}: {str(e)}")
                var_results['adf'] = None
            
            try:
                kpss_result = kpss(data_series, regression='c', nlags="auto")
                var_results['kpss_statistic'] = kpss_result[0]
                var_results['kpss_pvalue'] = kpss_result[1]
            except Exception as e:
                print(f"Warning: KPSS test failed for {var}: {str(e)}")
                var_results['kpss'] = None
            
            if self.plotting:
                plt.figure(figsize=(12, 4))
                plt.plot(data_series.index, data_series.values, marker='o', linestyle='-')
                plt.title(f"Time Series Plot of {var}")
                plt.xlabel("Time")
                plt.ylabel(var)
                self.save_plot(f"{var}_timeseries.png")
                
                # Plot ACF and PACF
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                plot_acf(data_series, ax=ax[0])
                ax[0].set_title(f"ACF of {var}")
                plot_pacf(data_series, ax=ax[1])
                ax[1].set_title(f"PACF of {var}")
                plt.tight_layout()
                self.save_plot(f"{var}_acf_pacf.png")
            
            stationarity_results[var] = var_results
        
        return stationarity_results
    
    def check_heteroscedasticity(self) -> Dict:
        """
        Check for heteroscedasticity in time series data using Engle's ARCH test.
        
        Returns:
            Dict: A dictionary with ARCH test results for each variable.
        """
        hetero_results = {}
        
        for var in self.variables:
            data_series = pd.to_numeric(self.data[var], errors='coerce').dropna()
            if len(data_series) < 10:
                print(f"Warning: Not enough observations for heteroscedasticity test for {var}")
                continue
            
            try:
                # Remove mean to get residual-like series
                series_demeaned = data_series - data_series.mean()
                arch_result = het_arch(series_demeaned)
                hetero_results[var] = {
                    'LM_statistic': arch_result[0],
                    'LM_pvalue': arch_result[1],
                    'F_statistic': arch_result[2],
                    'F_pvalue': arch_result[3]
                }
            except Exception as e:
                print(f"Warning: ARCH test failed for {var}: {str(e)}")
                hetero_results[var] = None
        
        return hetero_results

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
            results (Dict): Results from validation checks.
            
        Returns:
            str: Formatted summary text.
        """
        summary = ["=== Data Validation Summary ===\n"]
        
        # Distributions
        if 'distributions' in results:
            summary.append("\n1. Distribution Summary:")
            for var, var_results in results['distributions'].items():
                if var_results:
                    summary.append(f"\n{var} ({var_results['type']}):")
                    if var_results['type'] == 'numeric':
                        stats_dict = var_results['basic_stats']
                        summary.append(f"  Mean: {stats_dict['mean']:.2f}")
                        summary.append(f"  Median: {stats_dict['median']:.2f}")
                        summary.append(f"  Std: {stats_dict['std']:.2f}")
                        summary.append(f"  Skewness: {stats_dict['skewness']:.2f}")
                        summary.append(f"  Kurtosis: {stats_dict['kurtosis']:.2f}")
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
            summary.append("\n2. Group Size Analysis:")
            for var, size_results in results['group_sizes'].items():
                if size_results:
                    summary.append(f"\n{var}:")
                    summary.append(f"  Min group size: {size_results['min_size']}")
                    summary.append(f"  Max group size: {size_results['max_size']}")
                    summary.append(f"  Max/Min ratio: {size_results['ratio']:.2f}")
                    summary.append(f"  Balance status: {'Balanced' if size_results['balanced'] else 'Imbalanced'}")
                    
                    summary.append("\n  Group counts:")
                    for group, count in size_results['counts'].items():
                        summary.append(f"    {group}: {count}")
        
        # Stationarity
        if 'stationarity' in results:
            summary.append("\n3. Stationarity Test Results:")
            for var, var_results in results['stationarity'].items():
                if var_results:
                    summary.append(f"{var}: ADF p-value = {var_results.get('adf_pvalue', 'N/A'):.4e}, "
                                   f"KPSS p-value = {var_results.get('kpss_pvalue', 'N/A'):.4e}")
        
        # Heteroscedasticity
        if 'heteroscedasticity' in results:
            summary.append("\n4. Heteroscedasticity Test Results:")
            for var, var_results in results['heteroscedasticity'].items():
                if var_results:
                    summary.append(f"{var}: ARCH LM p-value = {var_results.get('LM_pvalue', 'N/A'):.4e}, "
                                   f"F-test p-value = {var_results.get('F_pvalue', 'N/A'):.4e}")
        
        # Multicollinearity
        if 'multicollinearity' in results and results['multicollinearity'] is not None:
            summary.append("\n5. Multicollinearity Analysis:")
            
            # Handle correlation matrix
            if 'correlation_matrix' in results['multicollinearity']:
                try:
                    corr_data = results['multicollinearity']['correlation_matrix']
                    if isinstance(corr_data, dict) and all(isinstance(v, dict) for v in corr_data.values()):
                        summary.append("  Correlation matrix analysis:")
                        high_corr = []
                        variables = list(corr_data.keys())
                        for i, var1 in enumerate(variables):
                            for j, var2 in enumerate(variables):
                                if i < j:
                                    try:
                                        corr = corr_data[var1].get(var2, 0)
                                        if abs(corr) > 0.7:
                                            high_corr.append(f"  High correlation between {var1} and {var2}: {corr:.2f}")
                                    except Exception as e:
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
                    summary.append(f"\n  VIF analysis: {vif_scores}")
            
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
            checks (List[str]): List of checks to run.
            **kwargs: Additional arguments for specific checks.
        """
        output_capture = OutputCapture(self.output_dir, 'data_validation_results.txt')
        sys.stdout = output_capture

        try:
            available_checks = {
                'distributions': self.check_distributions,
                'group_sizes': self.check_group_sizes,
                'multicollinearity': self.check_multicollinearity,
                'stationarity': self.check_stationarity,
                'heteroscedasticity': self.check_heteroscedasticity
            }

            print("=== Data Validation Results ===")
            
            results = {}

            for check in checks:
                print(f"DEBUG: Processing check: {check}")
                if check in available_checks:
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