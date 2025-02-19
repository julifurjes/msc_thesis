import pandas as pd
import numpy as np
from semopy import Model
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import graphviz

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.save_output import OutputCapture, get_output_dir 

class MenopauseLongitudinalAnalysis:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path, low_memory=False)
        self.output_dir = get_output_dir('3_social_model', 'longitudinal')
        
        # Define variable groups
        self.social_vars = ['LISTEN', 'TAKETOM', 'NOTSMAR', 'PHYSPRO']
        self.emotional_vars = ['EMOCTDW', 'EMOACCO', 'EMOCARE']
        self.social_health_vars = ['INTERFR', 'SOCIAL']
        self.cognitive_vars = ['TOTIDE1', 'TOTIDE2']
        self.symptom_vars = ['NITESWE', 'BOTCLDS', 'IRRITAB', 'MOODCHG']
        self.control_vars = ['STATUS']
        self.socioeco_vars = ['INCOME', 'HOW_HAR', 'BCINCML', 'DEGREE']
        
    def preprocess_longitudinal_data(self):
        """Prepare data for longitudinal analysis"""
        # Convert variables to numeric
        all_vars = (self.social_vars + self.emotional_vars + self.social_health_vars + 
                   self.cognitive_vars + self.symptom_vars + self.control_vars + 
                   self.socioeco_vars)
        
        for col in all_vars:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        # Recode time to start at 0
        self.data['time'] = self.data['VISIT'] - self.data['VISIT'].min()
        
        # Sort data by subject and time
        self.data = self.data.sort_values(['SWANID', 'time'])
        
        # Standardize continuous variables within each time point
        continuous_vars = (self.social_vars + self.emotional_vars + self.social_health_vars + 
                         self.cognitive_vars + self.symptom_vars + self.socioeco_vars)
        
        for col in continuous_vars:
            self.data[col] = self.data.groupby('time')[col].transform(
                lambda x: (x - x.mean()) / x.std()
            )
    
    def create_lgcm_syntax(self, outcome_var):
        """Create LGCM syntax for a single outcome variable"""
        # Get unique time points
        time_points = sorted(self.data['time'].unique())
        max_time = max(time_points)
        
        # Create variable names for each time point
        var_names = [f"{outcome_var}_t{int(t)}" for t in time_points]
        
        # Build model syntax
        syntax = f"""
        # Growth Factors
        i_{outcome_var} =~ 1*{var_names[0]}"""
        
        # Add loadings for intercept factor
        for var in var_names[1:]:
            syntax += f" + 1*{var}"
        
        # Add slope factor with linear time scores
        syntax += f"\ns_{outcome_var} =~ 0*{var_names[0]}"
        for t, var in zip(time_points[1:], var_names[1:]):
            syntax += f" + {t}*{var}"
        
        # Add mean structure
        syntax += f"""
        # Growth Factor Mean Structure
        i_{outcome_var} ~ mean*1
        s_{outcome_var} ~ mean*1
        
        # Growth Factor Variances and Covariance
        i_{outcome_var} ~~ var*i_{outcome_var}
        s_{outcome_var} ~~ var*s_{outcome_var}
        i_{outcome_var} ~~ cov*s_{outcome_var}
        """
        
        # Add residual variances (constrained equal across time)
        syntax += f"\n# Residual Variances"
        for var in var_names:
            syntax += f"\n{var} ~~ res*{var}"
        
        return syntax, var_names
    
    def create_parallel_process_model(self):
        """Create parallel process LGCM for multiple outcomes"""
        # Create individual LGCMs
        model_parts = []
        all_var_names = {}
        
        # Add LGCM for cognitive outcomes
        for var in self.cognitive_vars:
            syntax, var_names = self.create_lgcm_syntax(var)
            model_parts.append(syntax)
            all_var_names[var] = var_names
        
        # Add LGCM for social support (using first social var as example)
        syntax, var_names = self.create_lgcm_syntax(self.social_vars[0])
        model_parts.append(syntax)
        all_var_names[self.social_vars[0]] = var_names
        
        # Add LGCM for emotional wellbeing (using first emotional var as example)
        syntax, var_names = self.create_lgcm_syntax(self.emotional_vars[0])
        model_parts.append(syntax)
        all_var_names[self.emotional_vars[0]] = var_names
        
        # Combine all parts
        full_model = "\n".join(model_parts)
        
        # Add cross-domain relationships
        full_model += f"""
        # Cross-domain structural paths
        i_TOTIDE1 ~ p1*i_{self.social_vars[0]} + p2*i_{self.emotional_vars[0]}
        s_TOTIDE1 ~ p3*s_{self.social_vars[0]} + p4*s_{self.emotional_vars[0]}
        
        i_TOTIDE2 ~ p5*i_{self.social_vars[0]} + p6*i_{self.emotional_vars[0]}
        s_TOTIDE2 ~ p7*s_{self.social_vars[0]} + p8*s_{self.emotional_vars[0]}
        """
        
        return full_model, all_var_names
    
    def reshape_data_for_lgcm(self):
        """Reshape data from long to wide format for LGCM"""
        # Create time-specific variables
        wide_data = []
        
        variables = (self.cognitive_vars + 
                    [self.social_vars[0]] +  # Using first social var
                    [self.emotional_vars[0]])  # Using first emotional var
        
        for var in variables:
            # Pivot to wide format
            var_wide = self.data.pivot(
                index='SWANID',
                columns='time',
                values=var
            )
            # Rename columns to include variable name
            var_wide.columns = [f"{var}_t{int(t)}" for t in var_wide.columns]
            wide_data.append(var_wide)
        
        # Add constant column for means
        constant = pd.DataFrame({'1': 1}, index=wide_data[0].index)
        wide_data.append(constant)
        
        # Combine all wide format data
        wide_data = pd.concat(wide_data, axis=1)
        
        return wide_data
    
    def fit_lgcm(self):
        """Fit the parallel process LGCM"""
        try:
            # Reshape data to wide format
            wide_data = self.reshape_data_for_lgcm()
            
            # Create model syntax
            model_syntax, var_names = self.create_parallel_process_model()
            
            # Save model syntax for debugging
            with open(os.path.join(self.output_dir, 'model_syntax.txt'), 'w') as f:
                f.write(model_syntax)
            
            # Create and fit model
            model = Model(model_syntax)
            results = model.fit(wide_data)
            
            # Get parameter estimates
            params = model.inspect(std_est=True)
            
            # Save results
            with open(os.path.join(self.output_dir, 'lgcm_results.txt'), 'w') as f:
                f.write("Parallel Process LGCM Results\n")
                f.write("===========================\n\n")
                f.write("Parameter Estimates:\n")
                f.write(str(params))
                
            return model, results, params, var_names
            
        except Exception as e:
            print(f"Error fitting LGCM: {str(e)}")
            return None, None, None, None
    
    def plot_growth_trajectories(self, model, var_names):
        """Plot estimated growth trajectories"""
        if model is not None and model.Estimates is not None:
            try:
                time_points = sorted(self.data['time'].unique())
                
                for outcome in (self.cognitive_vars + 
                              [self.social_vars[0]] + 
                              [self.emotional_vars[0]]):
                    # Get growth parameters
                    i_mean = model.Estimates[f'i_{outcome}~mean']
                    s_mean = model.Estimates[f's_{outcome}~mean']
                    
                    # Calculate predicted trajectory
                    predicted = [i_mean + s_mean*t for t in time_points]
                    
                    # Plot
                    plt.figure(figsize=(10, 6))
                    plt.plot(time_points, predicted, 'b-', label='Estimated Trajectory')
                    plt.scatter(
                        self.data['time'],
                        self.data[outcome],
                        alpha=0.1,
                        color='gray',
                        label='Observed Values'
                    )
                    plt.title(f'Growth Trajectory for {outcome}')
                    plt.xlabel('Time')
                    plt.ylabel('Standardized Score')
                    plt.legend()
                    plt.savefig(
                        os.path.join(self.output_dir, f'growth_trajectory_{outcome}.png')
                    )
                    plt.close()
                    
            except Exception as e:
                print(f"Error plotting trajectories: {str(e)}")
    
    def run_complete_analysis(self):
        """Run complete longitudinal analysis"""
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("Preprocessing longitudinal data...")
        self.preprocess_longitudinal_data()
        
        print("Fitting parallel process LGCM...")
        model, results, params, var_names = self.fit_lgcm()
        
        if model is not None:
            print("Creating growth trajectory plots...")
            self.plot_growth_trajectories(model, var_names)
            
            print("\nModel successfully fitted!")
            print(f"Number of subjects: {len(self.data['SWANID'].unique())}")
            print(f"Number of time points: {len(self.data['time'].unique())}")
            print(f"Number of parameters estimated: {len(params)}")
            
            return {
                'model': model,
                'results': results,
                'params': params,
                'var_names': var_names
            }
        else:
            print("Model fitting failed. Please check the data and model specification.")
            return None

if __name__ == "__main__":
    # Initialize and run analysis
    analysis = MenopauseLongitudinalAnalysis("processed_combined_data.csv")
    results = analysis.run_complete_analysis()