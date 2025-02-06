import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import gee
from statsmodels.genmod.cov_struct import Exchangeable
import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.save_output import OutputCapture, get_output_dir 

class MenopauseAnalysis:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path, low_memory=False)
        self.filter_status()
        self.gee_results = {}
        self.output_dir = get_output_dir('1_stages_model', 'within-subjects') 
    
    def filter_status(self):
        """Filter the dataset to include only subjects with STATUS 2-5."""
        self.data['STATUS'] = pd.to_numeric(self.data['STATUS'], errors='coerce')
        self.data = self.data[self.data['STATUS'].between(2, 5)]
        
        # Map STATUS to more descriptive labels
        status_map = {
            2: 'Post-menopause',
            3: 'Late Peri',
            4: 'Early Peri',
            5: 'Pre-menopause'
        }
        self.data['STATUS_Label'] = self.data['STATUS'].map(status_map)
        
        # Create a categorical type with proper order for plotting
        status_order = ['Pre-menopause', 'Early Peri', 'Late Peri', 'Post-menopause']
        self.data['STATUS_Label'] = pd.Categorical(
            self.data['STATUS_Label'],
            categories=status_order,
            ordered=True
        )
    
    def run_gee_analysis(self, covariates=None):
        """
        Run GEE analysis for each cognitive variable, using STATUS as a categorical variable.
        """
        cognitive_vars = ['TOTIDE1', 'TOTIDE2', 'NERVES', 'SAD', 'FEARFULA']
        
        # Ensure numeric types for cognitive variables
        for var in cognitive_vars:
            self.data[var] = pd.to_numeric(self.data[var], errors='coerce')
        
        # Prepare covariates string for formula
        if covariates is None:
            covariates = []
        covariate_str = ' + '.join(covariates) if covariates else ''
        
        # Dictionary to store results
        self.gee_results = {}
        
        for outcome in cognitive_vars:
            # Create formula using STATUS as categorical variable
            if covariate_str:
                formula = f"{outcome} ~ C(STATUS_Label) + {covariate_str}"
            else:
                formula = f"{outcome} ~ C(STATUS_Label)"
            
            # Fit GEE model
            try:
                model = gee(
                    formula=formula,
                    groups="SWANID",
                    data=self.data.dropna(subset=[outcome] + covariates),
                    cov_struct=Exchangeable(),
                    family=sm.families.Gaussian()
                )
                
                results = model.fit()
                self.gee_results[outcome] = results
                
                # Print detailed results
                print(f"\nGEE Results for {outcome}")
                print("=" * 50)
                print(results.summary())
                
            except Exception as e:
                print(f"Error in GEE analysis for {outcome}: {str(e)}")
    
    def run_complete_analysis(self, covariates=None):
        """Run the complete analysis pipeline including GEE analysis."""
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up output capture
        output_capture = OutputCapture(self.output_dir)
        sys.stdout = output_capture

        try:
            print("\nRunning GEE analysis...")
            self.run_gee_analysis(covariates=covariates)
        
        finally:
            sys.stdout = output_capture.terminal
            output_capture.close()

if __name__ == "__main__":
    # Initialize analysis with your data file
    analysis = MenopauseAnalysis("processed_combined_data.csv")
    
    # Run the complete analysis with optional covariates
    covariates = ['AGE']
    analysis.run_complete_analysis(covariates=covariates)