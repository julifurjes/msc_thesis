import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import boxcox
import os

class FurtherChecks:
    def __init__(self):
        """
        Initialize the FurtherChecks object.
        """

    def examine_distributions(self, data, variables_to_check, output_dir):
        """
        Examine the distributions of outcome variables and apply transformations if needed.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing the variables to examine
        variables_to_check : list
            List of variable column names to check and potentially transform
        output_dir : str
            Directory path where output plots should be saved
        
        Returns:
        --------
        tuple : (pandas.DataFrame, dict)
            Tuple containing:
            - Updated DataFrame with transformed variables added
            - Dictionary with transformation information for each variable
        """
        print("\nExamining outcome distributions and testing transformations...")
        
        # Create a copy of the input data to avoid modifying the original
        data = data.copy()
        
        # Create directory for distribution plots
        dist_dir = os.path.join(output_dir, 'distribution_plots')
        os.makedirs(dist_dir, exist_ok=True)
        
        # Dictionary to store transformation decisions
        transformed_vars = {}
        
        for var in variables_to_check:
            # Convert to numeric
            data[var] = pd.to_numeric(data[var], errors='coerce')
            
            # Get clean data for this variable (non-missing values)
            clean_data = data[var].dropna()
            
            if len(clean_data) == 0:
                print(f"Warning: No valid data for {var}")
                continue
            
            # Calculate distribution statistics
            n = len(clean_data)
            mean = clean_data.mean()
            median = clean_data.median()
            skewness = stats.skew(clean_data)
            kurtosis = stats.kurtosis(clean_data)
            
            # Normality test
            _, shapiro_p = stats.shapiro(clean_data.sample(min(n, 5000)) if n > 5000 else clean_data)
            
            print(f"\n{var} Distribution Statistics:")
            print(f"N = {n}")
            print(f"Mean = {mean:.2f}")
            print(f"Median = {median:.2f}")
            print(f"Skewness = {skewness:.3f}")
            print(f"Kurtosis = {kurtosis:.3f}")
            print(f"Shapiro-Wilk p-value = {shapiro_p:.4f}")
            
            # Create a figure for the plots
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
            
            # Plot original distribution histogram with density curve
            sns.histplot(clean_data, kde=True, ax=ax1)
            ax1.set_title(f"Original Distribution of {var}")
            ax1.set_ylabel("Frequency")
            ax1.text(0.05, 0.95, f"Skewness: {skewness:.3f}\nKurtosis: {kurtosis:.3f}\np-value: {shapiro_p:.4f}",
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Decide if transformation is needed
            needs_transform = False
            transform_method = None
            transformed_data = None
            
            if abs(skewness) > 1 or shapiro_p < 0.05:
                needs_transform = True
                
                # Try log transformation for positive data
                if clean_data.min() >= 0:
                    log_data = np.log1p(clean_data)
                    log_skewness = stats.skew(log_data)
                    log_kurtosis = stats.kurtosis(log_data)
                    _, log_p = stats.shapiro(log_data.sample(min(n, 5000)) if n > 5000 else log_data)
                    
                    # Plot log-transformed distribution
                    sns.histplot(log_data, kde=True, ax=ax2)
                    ax2.set_title(f"Log-Transformed {var}")
                    ax2.text(0.05, 0.95, f"Skewness: {log_skewness:.3f}\nKurtosis: {log_kurtosis:.3f}\np-value: {log_p:.4f}",
                            transform=ax2.transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    # Try Box-Cox transformation
                    if clean_data.min() > 0:
                        try:
                            # Find optimal lambda value for Box-Cox
                            boxcox_data, lambda_val = boxcox(clean_data)
                            boxcox_skewness = stats.skew(boxcox_data)
                            boxcox_kurtosis = stats.kurtosis(boxcox_data)
                            if len(boxcox_data) > 5000:
                                boxcox_series = pd.Series(boxcox_data)
                                boxcox_sample = boxcox_series.sample(5000)
                                _, boxcox_p = stats.shapiro(boxcox_sample)

                            else:
                                _, boxcox_p = stats.shapiro(boxcox_data)
                            
                            # Plot Box-Cox transformed distribution
                            sns.histplot(boxcox_data, kde=True, ax=ax3)
                            ax3.set_title(f"Box-Cox Transformed {var} (λ = {lambda_val:.3f})")
                            ax3.text(0.05, 0.95, f"Skewness: {boxcox_skewness:.3f}\nKurtosis: {boxcox_kurtosis:.3f}\np-value: {boxcox_p:.4f}",
                                    transform=ax3.transAxes, verticalalignment='top',
                                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                            
                            # Compare transformations and choose the best one
                            transforms = {
                                'original': (abs(skewness), shapiro_p, None, clean_data),
                                'log': (abs(log_skewness), log_p, 'log', log_data),
                                'boxcox': (abs(boxcox_skewness), boxcox_p, 'boxcox', boxcox_data)
                            }
                            
                            # Sort by skewness first, then by p-value (higher is better)
                            best_transform = min(transforms.items(), key=lambda x: (x[1][0], -x[1][1]))
                            transform_method = best_transform[1][2]
                            transformed_data = best_transform[1][3]
                            
                            print(f"Best transformation for {var}: {best_transform[0]}")
                            print(f"  Skewness: {best_transform[1][0]:.3f}, p-value: {best_transform[1][1]:.4f}")
                            
                        except Exception as e:
                            print(f"Box-Cox transformation failed for {var}: {str(e)}")
                            # If Box-Cox fails, just use log transformation
                            if abs(log_skewness) < abs(skewness):
                                transform_method = 'log'
                                transformed_data = log_data
                                ax3.text(0.5, 0.5, "Box-Cox transformation failed",
                                        ha='center', va='center')
                            else:
                                transform_method = None  # No transformation
                                ax3.text(0.5, 0.5, "No suitable transformation found",
                                        ha='center', va='center')
                else:
                    print(f"{var} contains non-positive values, skipping log and Box-Cox transformations")
                    ax2.text(0.5, 0.5, "Cannot apply log to non-positive values",
                            ha='center', va='center')
                    ax3.text(0.5, 0.5, "Cannot apply Box-Cox to non-positive values",
                            ha='center', va='center')
            else:
                print(f"{var} distribution is acceptably normal, no transformation needed")
                ax2.text(0.5, 0.5, "No transformation needed",
                        ha='center', va='center')
                ax3.text(0.5, 0.5, "No transformation needed",
                        ha='center', va='center')
            
            # Apply transformation to the dataset if needed
            if transform_method:
                if transform_method == 'log':
                    # Ensure all values are positive before applying log
                    min_val = data[var].min()
                    if min_val <= 0:
                        offset = abs(min_val) + 1  # Add offset to make all values positive
                        data[f'{var}_transformed'] = np.log1p(data[var] + offset)
                        print(f"Applied log transformation to {var} with offset {offset}")
                        transformed_vars[var] = {
                            'method': 'log',
                            'offset': offset,
                            'original_var': var
                        }
                    else:
                        data[f'{var}_transformed'] = np.log1p(data[var])
                        print(f"Applied log transformation to {var}")
                        transformed_vars[var] = {
                            'method': 'log',
                            'offset': 0,
                            'original_var': var
                        }
                elif transform_method == 'boxcox':
                    # For Box-Cox, we need to store the lambda value
                    lambda_val = stats.boxcox_normmax(clean_data)
                    # Use a small offset if needed to ensure positivity
                    min_val = data[var].min()
                    if min_val <= 0:
                        offset = abs(min_val) + 1
                        data[f'{var}_transformed'] = stats.boxcox(data[var] + offset, lambda_val)
                        print(f"Applied Box-Cox transformation to {var} with lambda={lambda_val:.3f} and offset {offset}")
                        transformed_vars[var] = {
                            'method': 'boxcox',
                            'lambda': lambda_val,
                            'offset': offset,
                            'original_var': var
                        }
                    else:
                        data[f'{var}_transformed'] = stats.boxcox(data[var], lambda_val)
                        print(f"Applied Box-Cox transformation to {var} with lambda={lambda_val:.3f}")
                        transformed_vars[var] = {
                            'method': 'boxcox',
                            'lambda': lambda_val,
                            'offset': 0,
                            'original_var': var
                        }
            
            # Save the plot
            plt.tight_layout()
            fig.savefig(os.path.join(dist_dir, f'{var}_distribution.png'), dpi=300)
            plt.close(fig)
            
        return data, transformed_vars