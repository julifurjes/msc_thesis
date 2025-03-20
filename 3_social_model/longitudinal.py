import pandas as pd
import numpy as np
import semopy
from semopy import Model
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

from helpers import SEMVisualizer, SEMAnalyzer

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.save_output import OutputCapture, get_output_dir

class MenopauseCognitionAnalysis:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path, low_memory=False)
        self.output_dir = get_output_dir('3_social_model', 'multilevel')
        
        # Define variable groups
        self.social_support_vars = ['LISTEN', 'TAKETOM', 'HELPSIC', 'CONFIDE']
        self.emotional_struggle_vars = ['EMOCTDW', 'EMOACCO', 'EMOCARE']
        self.social_struggle_vars = ['INTERFR', 'SOCIAL']
        self.cognitive_vars = ['TOTIDE1', 'TOTIDE2']
        self.symptom_vars = ['HOTFLAS', 'NUMHOTF', 'BOTHOTF', 'NITESWE', 'NUMNITS', 'BOTNITS',
                            'COLDSWE', 'NUMCLDS', 'BOTCLDS', 'STIFF', 'IRRITAB', 'MOODCHG']
        self.control_vars = ['STATUS', 'LANGCOG']
        self.socioeco_vars = ['INCOME', 'HOW_HAR', 'BCINCML', 'DEGREE']
        
    def preprocess_data(self):
        """Prepare data for SEM analysis with repeated measures"""
        # Convert all variables to numeric
        relevant_vars = (self.social_support_vars + self.emotional_struggle_vars + self.social_struggle_vars + 
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
        
        # Convert SWANID to string to ensure it's treated as categorical
        self.data['SWANID'] = self.data['SWANID'].astype(str)
        
        # Standardize all continuous variables to help with convergence
        for var in self.social_support_vars + self.emotional_struggle_vars + self.social_struggle_vars + self.cognitive_vars + self.symptom_vars:
            self.data[var] = (self.data[var] - self.data[var].mean()) / self.data[var].std()
            
        print(f"Final dataset has {len(self.data)} observations from {self.data['SWANID'].nunique()} subjects")
        
    def create_measurement_model(self):
        """Define a simplified model that should have better chances of converging"""
        model_syntax = """
        # Measurement model
        social_support =~ LISTEN + TAKETOM + HELPSIC + CONFIDE
        emotional_struggle =~ EMOCTDW + EMOACCO + EMOCARE
        social_struggle =~ INTERFR + SOCIAL
        cognitive =~ TOTIDE1 + TOTIDE2
        symptoms =~ NITESWE + BOTCLDS + IRRITAB + MOODCHG

        # Simplified structural model: Main relationships
        cognitive ~ social_support + emotional_struggle + social_struggle + time_centered + STATUS + LANGCOG
        emotional_struggle ~ social_support + symptoms + time_centered + STATUS + LANGCOG 
        social_struggle ~ social_support + emotional_struggle + symptoms + time_centered + STATUS + LANGCOG

        # Correlations between exogenous variables
        social_support ~~ symptoms
        STATUS ~~ LANGCOG
        """
        return model_syntax
    
    def fit_sem(self):
        """Fit the SEM model with a simpler approach for within-subject data"""
        try:
            model_syntax = self.create_measurement_model()
            
            # Create model
            model = Model(model_syntax)
            
            # Standard fit without any special options
            results = model.fit(self.data)
            
            # Include cluster information in the output
            print("\nCluster Information:")
            cluster_counts = self.data['SWANID'].value_counts()
            print(f"Number of clusters (subjects): {len(cluster_counts)}")
            print(f"Average observations per subject: {cluster_counts.mean():.2f}")
            print(f"Min observations per subject: {cluster_counts.min()}")
            print(f"Max observations per subject: {cluster_counts.max()}")
            
            params = model.inspect(std_est=True)
            
            print("\nSEM Results:")
            print("=" * 50)
            print("\nModel Information:")
            print(f"Number of observations: {len(self.data)}")
            print(f"Number of parameters: {len(params)}")
            print("\nParameter Estimates:")
            print(params.to_string())
            
            return model, results, params
            
        except Exception as e:
            print(f"Error fitting model: {str(e)}")
            return None, None, None

    def run_complete_analysis(self):
        """Run within-subject SEM analysis with a simplified approach"""
        os.makedirs(self.output_dir, exist_ok=True)

        # Set up output capture
        output_capture = OutputCapture(self.output_dir)
        sys.stdout = output_capture

        try:
            print("Preprocessing data...")
            self.preprocess_data()
            
            # Print basic data information
            print(f"\nData summary:")
            print(f"Total observations: {len(self.data)}")
            print(f"Unique subjects: {self.data['SWANID'].nunique()}")
            print(f"Visits per subject: Min={self.data.groupby('SWANID').size().min()}, " +
                  f"Max={self.data.groupby('SWANID').size().max()}, " +
                  f"Mean={self.data.groupby('SWANID').size().mean():.2f}")
            
            # Show correlations between main variables
            print("\nCorrelations between key variables:")
            key_vars = ['LISTEN', 'EMOCARE', 'INTERFR', 'TOTIDE1', 'MOODCHG', 'time_centered']
            print(self.data[key_vars].corr().round(3).to_string())
            
            print("\nFitting SEM model with within-subject variability...")
            model, results, params = self.fit_sem()
            
            if model is not None:
                # Create analyzer for additional diagnostics
                analyzer = SEMAnalyzer(model, self.data, params)
                
                # Get and print fit indices
                print("\nModel Fit Statistics:")
                print(analyzer.format_fit_indices())
                
                # Create correlation heatmaps
                print("\nGenerating correlation heatmaps...")
                fig = analyzer.create_correlation_heatmaps()
                heatmap_path = os.path.join(self.output_dir, 'correlation_heatmaps.png')
                fig.savefig(heatmap_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                print(f"Correlation heatmaps saved to: {heatmap_path}")

                print("Generating path diagrams...")
                visualizer = SEMVisualizer(params)
                visualizer.save_visualizations(self.output_dir)

                # Filter parameter estimates to focus on the important ones
                structural_params = params[(params['op'] == '~') & 
                                          (params['lval'].isin(['cognitive', 'emotional', 'social_health']))]
                
                print("\nStructural Model Parameter Estimates:")
                print(structural_params.to_string())
                
                print("\nMeasurement Model Parameter Estimates:")
                measurement_params = params[params['op'] == '=~']
                print(measurement_params.to_string())

                print("\nAnalysis completed successfully!")
                print(f"Results saved to: {self.output_dir}")
                print("\nVisualization files have been created. Open the HTML files in a web browser to view the interactive diagrams.")

                return {
                    'model': model,
                    'results': results,
                    'params': params
                }
        
        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            return None
            
        finally:
            sys.stdout = output_capture.terminal
            output_capture.close()

if __name__ == "__main__":
    # Initialize and run analysis
    analysis = MenopauseCognitionAnalysis("processed_combined_data.csv")
    results = analysis.run_complete_analysis()