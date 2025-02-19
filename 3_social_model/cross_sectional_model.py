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

class MenopauseCognitionAnalysis:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path, low_memory=False)
        self.output_dir = get_output_dir('3_social_model', 'cross_sectional')
        
        # Define variable groups from original analysis
        self.social_vars = ['LISTEN', 'TAKETOM', 'NOTSMAR', 'PHYSPRO']
        self.emotional_vars = ['EMOCTDW', 'EMOACCO', 'EMOCARE']
        self.social_health_vars = ['INTERFR', 'SOCIAL']
        self.cognitive_vars = ['TOTIDE1', 'TOTIDE2']
        self.symptom_vars = ['NITESWE', 'BOTCLDS', 'IRRITAB', 'MOODCHG']
        self.control_vars = ['STATUS']
        self.socioeco_vars = ['INCOME', 'HOW_HAR', 'BCINCML', 'DEGREE']
        
    def preprocess_data(self):
        """Prepare data for SEM analysis"""
        # Take the first observation for each participant for cross-sectional analysis
        self.data = self.data.sort_values('VISIT').groupby('SWANID').first().reset_index()
        
        # Convert all variables to numeric
        all_vars = (self.social_vars + self.emotional_vars + self.social_health_vars + 
                   self.cognitive_vars + self.symptom_vars + self.control_vars + 
                   self.socioeco_vars)
        
        for col in all_vars:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        # Drop rows with missing values
        self.data = self.data.dropna(subset=all_vars)
        
        # Standardize continuous variables (excluding categorical controls if any)
        continuous_vars = (self.social_vars + self.emotional_vars + self.social_health_vars + 
                         self.cognitive_vars + self.symptom_vars + self.socioeco_vars)
        
        for col in continuous_vars:
            self.data[col] = (self.data[col] - self.data[col].mean()) / self.data[col].std()
        
    def create_measurement_model(self):
        """Define the measurement model for latent variables with control variables"""
        model_syntax = """
        # Measurement model
        # Social support latent variable
        social =~ LISTEN + TAKETOM + NOTSMAR + PHYSPRO
        
        # Emotional wellbeing latent variable
        emotional =~ EMOCTDW + EMOACCO + EMOCARE
        
        # Social health latent variable
        social_health =~ INTERFR + SOCIAL
        
        # Cognitive function latent variable
        cognitive =~ TOTIDE1 + TOTIDE2
        
        # Menopause symptoms latent variable
        symptoms =~ NITESWE + BOTCLDS + IRRITAB + MOODCHG
        
        # Structural model with control variables
        # Main relationships
        cognitive ~ social + emotional + symptoms
        emotional ~ social + symptoms
        social_health ~ social + emotional + symptoms
        
        # Control variables affecting endogenous variables
        cognitive ~ STATUS + INCOME + HOW_HAR + BCINCML + DEGREE
        emotional ~ STATUS + INCOME + HOW_HAR + BCINCML + DEGREE
        social_health ~ STATUS + INCOME + HOW_HAR + BCINCML + DEGREE
        
        # Allow correlations between control variables
        STATUS ~~ INCOME + HOW_HAR + BCINCML + DEGREE
        INCOME ~~ HOW_HAR + BCINCML + DEGREE
        HOW_HAR ~~ BCINCML + DEGREE
        BCINCML ~~ DEGREE
        """
        return model_syntax
    
    def fit_sem(self):
        """Fit the SEM model and return results"""
        model_syntax = self.create_measurement_model()
        model = Model(model_syntax)
        
        try:
            # Fit the model
            results = model.fit(self.data)
            
            # Get parameter estimates with standardized estimates
            params = model.inspect(std_est=True)
            
            # Save results
            with open(os.path.join(self.output_dir, 'sem_results.txt'), 'w') as f:
                f.write("Parameter Estimates (with standardized estimates):\n")
                f.write(str(params))
                f.write("\n\nModel Information:\n")
                f.write(f"Number of observations: {len(self.data)}\n")
                
                # Write parameter estimates summary
                f.write("\nParameter Estimates Summary:\n")
                f.write(f"Total number of parameters: {len(params)}\n")
                
                # Measurement model parameters
                f.write("\nMeasurement Model:\n")
                measurement_params = params[params['op'] == '=~']
                f.write(str(measurement_params))
                
                # Structural model parameters (main relationships)
                f.write("\nStructural Model (Main Relationships):\n")
                structural_params = params[
                    (params['op'] == '~') & 
                    (~params['rval'].isin(self.control_vars + self.socioeco_vars))
                ]
                f.write(str(structural_params))
                
                # Control variable effects
                f.write("\nControl Variable Effects:\n")
                control_params = params[
                    (params['op'] == '~') & 
                    (params['rval'].isin(self.control_vars + self.socioeco_vars))
                ]
                f.write(str(control_params))
                
            return model, results
            
        except Exception as e:
            print(f"Error fitting model: {str(e)}")
            return None, None
    
    def plot_path_diagram(self, model):
        """Create and save path diagram using graphviz"""
        if model is not None:
            try:
                # Create a graphviz graph
                dot = graphviz.Digraph(comment='SEM Path Diagram')
                dot.attr(rankdir='LR')
                
                # Add nodes for observed variables
                observed_vars = (self.social_vars + self.emotional_vars + 
                               self.social_health_vars + self.cognitive_vars + 
                               self.symptom_vars)
                for var in observed_vars:
                    dot.node(var, var, shape='box')
                
                # Add nodes for latent variables
                latent_vars = ['social', 'emotional', 'social_health', 'cognitive', 'symptoms']
                for var in latent_vars:
                    dot.node(var, var, shape='circle')
                
                # Get parameter estimates
                params = model.inspect(std_est=True)
                
                # Add edges based on model specification
                for _, row in params.iterrows():
                    if 'lval' in row and 'rval' in row:
                        # Add edge with coefficient as label
                        dot.edge(row['lval'], row['rval'], 
                               label=f"{row['Estimate']:.2f}")
                
                # Save the diagram
                dot.render(os.path.join(self.output_dir, 'path_diagram'), 
                         format='png', cleanup=True)
                
            except Exception as e:
                print(f"Error creating path diagram: {str(e)}")
    
    def run_complete_analysis(self):
        """Run complete SEM analysis"""
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        print("Preprocessing data...")
        self.preprocess_data()
        
        print("Fitting SEM model...")
        model, results = self.fit_sem()
        
        if model is not None:
            print("Creating path diagram...")
            self.plot_path_diagram(model)
            
            # Print basic model information
            print("\nModel successfully fitted!")
            print(f"Number of observations: {len(self.data)}")
            params = model.inspect(std_est=True)
            print(f"Number of parameters estimated: {len(params)}")
            
            return {
                'model': model,
                'results': results,
                'params': params
            }
        else:
            print("Model fitting failed. Please check the data and model specification.")
            return None

if __name__ == "__main__":
    # Initialize and run analysis
    analysis = MenopauseCognitionAnalysis("processed_combined_data.csv")
    results = analysis.run_complete_analysis()