import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.save_output import OutputCapture, get_output_dir

class MenopauseCognitionMixedModels:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path, low_memory=False)
        self.output_dir = get_output_dir('3_social_model', 'mixed_effects')
        
        # Define variable groups
        self.social_support_vars = ['LISTEN', 'TAKETOM', 'HELPSIC', 'CONFIDE']
        self.emotional_vars = ['EMOCTDW', 'EMOACCO', 'EMOCARE']
        self.social_health_vars = ['INTERFR', 'SOCIAL']
        self.cognitive_vars = ['TOTIDE1', 'TOTIDE2']
        self.symptom_vars = ['HOTFLAS', 'NUMHOTF', 'BOTHOTF', 'NITESWE', 'NUMNITS', 'BOTNITS',
                            'COLDSWE', 'NUMCLDS', 'BOTCLDS', 'STIFF', 'IRRITAB', 'MOODCHG']
        self.control_vars = ['STATUS', 'LANGCOG']
        self.socioeco_vars = ['INCOME', 'HOW_HAR', 'BCINCML', 'DEGREE']
        
    def preprocess_data(self):
        """Prepare data for mixed-effects model analysis"""
        # Convert all variables to numeric
        relevant_vars = (self.social_support_vars + self.emotional_vars + self.social_health_vars + 
                   self.cognitive_vars + self.symptom_vars + self.control_vars + 
                   self.socioeco_vars + ['SWANID', 'VISIT'])
        
        for col in relevant_vars:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

        # Filter dataset to keep only relevant columns
        self.data = self.data[relevant_vars]
        
        # Sort by subject ID and visit
        self.data = self.data.sort_values(['SWANID', 'VISIT'])
        
        # Add visit as a numeric time variable
        self.data['time'] = self.data['VISIT'].astype(float)
        
        # Center the time variable 
        self.data['time_centered'] = self.data['time'] - self.data['time'].mean()
        
        # Ensure SWANID is treated as a categorical variable
        self.data['SWANID'] = self.data['SWANID'].astype(str)
        
        # Create composite scores for each construct by averaging the indicator variables
        # This simplifies the analysis compared to latent variable modeling
        self.data['social_support'] = self.data[self.social_support_vars].mean(axis=1)
        self.data['emotional_wellbeing'] = self.data[self.emotional_vars].mean(axis=1)
        self.data['social_health'] = self.data[self.social_health_vars].mean(axis=1)
        self.data['cognitive_function'] = self.data[self.cognitive_vars].mean(axis=1)
        
        # For symptoms, we'll use a subset of the most relevant symptoms related to menopause
        key_symptoms = ['NITESWE', 'BOTCLDS', 'IRRITAB', 'MOODCHG']
        self.data['symptom_severity'] = self.data[key_symptoms].mean(axis=1)
        
        # Calculate Cronbach's alpha for each composite score to check reliability
        self._calculate_reliability()
        
        # Drop rows with missing values in the composite scores or key variables
        composite_vars = ['social_support', 'emotional_wellbeing', 'social_health', 
                         'cognitive_function', 'symptom_severity', 'time_centered'] + self.control_vars
        self.data = self.data.dropna(subset=composite_vars)
        
        print(f"Final dataset has {len(self.data)} observations from {self.data['SWANID'].nunique()} subjects")
        
    def _calculate_reliability(self):
        """Calculate Cronbach's alpha for all composite measures"""
        from sklearn.preprocessing import scale
        
        # Function to calculate Cronbach's alpha
        def cronbach_alpha(items):
            items = scale(items)  # standardize items
            items_count = items.shape[1]
            variance_sum = np.sum(np.var(items, axis=0))
            total_var = np.var(np.sum(items, axis=1))
            return (items_count / (items_count - 1)) * (1 - variance_sum / total_var)
        
        # Calculate alpha for each scale
        print("\nScale Reliability (Cronbach's alpha):")
        
        # Social Support
        alpha_social = cronbach_alpha(self.data[self.social_support_vars].dropna())
        print(f"Social Support: {alpha_social:.3f}")
        
        # Emotional Wellbeing
        alpha_emotional = cronbach_alpha(self.data[self.emotional_vars].dropna())
        print(f"Emotional Wellbeing: {alpha_emotional:.3f}")
        
        # Social Health
        alpha_social_health = cronbach_alpha(self.data[self.social_health_vars].dropna())
        print(f"Social Health: {alpha_social_health:.3f}")
        
        # Cognitive Function
        alpha_cognitive = cronbach_alpha(self.data[self.cognitive_vars].dropna())
        print(f"Cognitive Function: {alpha_cognitive:.3f}")
        
        # Symptoms (subset)
        key_symptoms = ['NITESWE', 'BOTCLDS', 'IRRITAB', 'MOODCHG']
        alpha_symptoms = cronbach_alpha(self.data[key_symptoms].dropna())
        print(f"Symptom Severity: {alpha_symptoms:.3f}")
    
    def fit_mixed_models(self):
        """Fit linear mixed-effects models for each outcome variable"""
        results = {}
        
        # Define the outcomes to model
        outcomes = {
            'cognitive_function': 'Cognitive Function',
            'emotional_wellbeing': 'Emotional Wellbeing',
            'social_health': 'Social Health'
        }
        
        for outcome_var, outcome_name in outcomes.items():
            print(f"\n\n{'='*50}")
            print(f"Modeling {outcome_name}")
            print(f"{'='*50}")
            
            # Create formula with fixed effects and random intercepts for subjects
            formula = (f"{outcome_var} ~ social_support + emotional_wellbeing + social_health + "
                      f"symptom_severity + time_centered + STATUS + LANGCOG")
            
            # Exclude the outcome variable from predictors if it appears there
            formula = self._adjust_formula(formula, outcome_var)
            
            # Fit the mixed-effects model
            try:
                print(f"\nFormula: {formula}")
                mixed_model = smf.mixedlm(
                    formula=formula,
                    data=self.data,
                    groups=self.data["SWANID"]
                )
                model_result = mixed_model.fit()
                
                # Store and print results
                results[outcome_var] = model_result
                print("\nModel Summary:")
                print(model_result.summary())
                
                # Calculate model fit statistics
                self._calculate_model_fit(model_result, outcome_var)
                
            except Exception as e:
                print(f"Error fitting model for {outcome_name}: {str(e)}")
        
        return results
    
    def _adjust_formula(self, formula, outcome_var):
        """Adjust formula to prevent including the outcome as a predictor"""
        for var in ['social_support', 'emotional_wellbeing', 'social_health']:
            if var == outcome_var:
                formula = formula.replace(f"{var} + ", "")
                formula = formula.replace(f"+ {var}", "")
        return formula
    
    def _calculate_model_fit(self, model_result, outcome_var):
        """Calculate and print model fit statistics"""
        # Get the actual values
        y_true = model_result.model.endog
        
        # Get the predicted values
        y_pred = model_result.fittedvalues
        
        # Calculate R-squared
        r_squared = model_result.rsquared
        
        # Calculate RMSE (Root Mean Squared Error)
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        
        # Calculate AIC and BIC
        aic = model_result.aic
        bic = model_result.bic
        
        # Calculate mean absolute error
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Print fit statistics
        print("\nModel Fit Statistics:")
        print(f"R-squared: {r_squared:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"AIC: {aic:.2f}")
        print(f"BIC: {bic:.2f}")
        
        # Create a scatter plot of observed vs. predicted values
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
        plt.xlabel('Observed Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Observed vs. Predicted Values for {outcome_var.replace("_", " ").title()}')
        plt.tight_layout()
        
        # Save the plot
        plot_filename = os.path.join(self.output_dir, f'{outcome_var}_predicted_vs_observed.png')
        plt.savefig(plot_filename, dpi=300)
        plt.close()
        
    def create_visualizations(self, results):
        """Generate visualizations for model results"""
        # Create forest plot for standardized coefficients
        self._create_coefficient_plot(results)
        
        # Create time trend plots
        self._create_time_trends()
        
    def _create_coefficient_plot(self, results):
        """Create a forest plot of standardized coefficients across models"""
        # Collect standardized coefficients from each model
        coef_data = []
        
        for outcome, result in results.items():
            # Get coefficients and standard errors
            params = result.params.drop('Intercept', errors='ignore')
            std_errors = result.bse.drop('Intercept', errors='ignore')
            
            # Calculate t-values and p-values
            t_values = params / std_errors
            p_values = [2 * (1 - stats.t.cdf(abs(t), result.df_resid)) for t in t_values]
            
            # Get variable names
            var_names = params.index.tolist()
            
            # Add to collection
            for var_name, coef, se, p in zip(var_names, params, std_errors, p_values):
                # Skip group random effect
                if var_name == 'Group Var':
                    continue
                    
                # Create nice variable name for display
                display_name = var_name.replace('_', ' ').title()
                
                # Add significance marker
                sig_marker = ''
                if p < 0.001:
                    sig_marker = '***'
                elif p < 0.01:
                    sig_marker = '**'
                elif p < 0.05:
                    sig_marker = '*'
                
                coef_data.append({
                    'Outcome': outcome.replace('_', ' ').title(),
                    'Predictor': display_name,
                    'Coefficient': coef,
                    'SE': se,
                    'P-Value': p,
                    'Significance': sig_marker
                })
        
        # Convert to DataFrame
        coef_df = pd.DataFrame(coef_data)
        
        # Plot coefficients
        plt.figure(figsize=(12, 8))
        g = sns.catplot(
            data=coef_df,
            x='Coefficient',
            y='Predictor',
            hue='Outcome',
            kind='point',
            height=8,
            aspect=1.5,
            join=False,
            errwidth=1.5,
            capsize=0.1
        )
        
        # Add vertical line at zero
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        
        # Add significance markers
        for i, row in coef_df.iterrows():
            if row['Significance']:
                plt.text(
                    row['Coefficient'] + row['SE'] * 1.2,
                    i,
                    row['Significance'],
                    ha='left',
                    va='center'
                )
                
        # Formatting
        plt.title('Coefficient Estimates with 95% Confidence Intervals', fontsize=16)
        plt.xlabel('Coefficient Estimate', fontsize=14)
        plt.ylabel('Predictor', fontsize=14)
        plt.tight_layout()
        
        # Save figure
        plot_path = os.path.join(self.output_dir, 'coefficient_forest_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create a table of coefficients
        coef_table = coef_df.pivot(index='Predictor', columns='Outcome', values='Coefficient')
        
        # Save table to CSV
        table_path = os.path.join(self.output_dir, 'coefficient_table.csv')
        coef_table.to_csv(table_path)
        
        print(f"\nCoefficient plot saved to: {plot_path}")
        print(f"Coefficient table saved to: {table_path}")
    
    def _create_time_trends(self):
        """Create plots showing time trends for key variables"""
        # Calculate mean values per visit
        visit_means = self.data.groupby('VISIT')[
            ['social_support', 'emotional_wellbeing', 'social_health', 
             'cognitive_function', 'symptom_severity']
        ].mean()
        
        # Calculate standard errors
        visit_se = self.data.groupby('VISIT')[
            ['social_support', 'emotional_wellbeing', 'social_health', 
             'cognitive_function', 'symptom_severity']
        ].sem()
        
        # Plot trends over time
        plt.figure(figsize=(12, 8))
        
        for i, variable in enumerate([
            'social_support', 'emotional_wellbeing', 'social_health', 
            'cognitive_function', 'symptom_severity'
        ]):
            plt.subplot(2, 3, i+1)
            
            # Plot mean with error bars
            plt.errorbar(
                visit_means.index,
                visit_means[variable],
                yerr=visit_se[variable],
                marker='o',
                linestyle='-',
                capsize=3
            )
            
            # Formatting
            plt.title(variable.replace('_', ' ').title())
            plt.xlabel('Visit')
            plt.ylabel('Mean Score')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        plot_path = os.path.join(self.output_dir, 'time_trends.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        print(f"Time trend plots saved to: {plot_path}")
        
    def run_complete_analysis(self):
        """Run the full mixed-effects analysis"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up output capture
        output_capture = OutputCapture(self.output_dir)
        sys.stdout = output_capture
        
        try:
            print("=" * 80)
            print("MENOPAUSE COGNITION MIXED-EFFECTS MODEL ANALYSIS")
            print("=" * 80)
            
            print("\nPreprocessing data...")
            self.preprocess_data()
            
            # Show descriptive statistics
            print("\nDescriptive Statistics for Composite Variables:")
            desc_stats = self.data[['social_support', 'emotional_wellbeing', 'social_health', 
                                   'cognitive_function', 'symptom_severity']].describe()
            print(desc_stats)
            
            # Show correlations
            print("\nCorrelation Matrix for Composite Variables:")
            corr_matrix = self.data[['social_support', 'emotional_wellbeing', 'social_health', 
                                    'cognitive_function', 'symptom_severity', 'time_centered']].corr()
            print(corr_matrix.round(3))
            
            # Create correlation heatmap
            plt.figure(figsize=(10, 8))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(
                corr_matrix,
                mask=mask,
                cmap='RdBu_r',
                annot=True,
                fmt='.2f',
                center=0,
                square=True,
                linewidths=.5
            )
            plt.title('Correlation Heatmap of Composite Variables', fontsize=16)
            plt.tight_layout()
            
            # Save correlation heatmap
            heatmap_path = os.path.join(self.output_dir, 'correlation_heatmap.png')
            plt.savefig(heatmap_path, dpi=300)
            plt.close()
            
            print(f"Correlation heatmap saved to: {heatmap_path}")
            
            # Fit mixed-effects models
            print("\nFitting mixed-effects models...")
            results = self.fit_mixed_models()
            
            # Create visualizations
            print("\nCreating visualizations...")
            self.create_visualizations(results)
            
            print("\nAnalysis completed successfully!")
            print(f"Results saved to: {self.output_dir}")
            
            return results
            
        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
        finally:
            sys.stdout = output_capture.terminal
            output_capture.close()

if __name__ == "__main__":
    # Initialize and run analysis
    analysis = MenopauseCognitionMixedModels("processed_combined_data.csv")
    results = analysis.run_complete_analysis()