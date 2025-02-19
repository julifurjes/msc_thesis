import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.save_output import OutputCapture, get_output_dir 

class MenopauseSocialCognitionModel:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path, low_memory=False)
        self.social_vars = ['LISTEN', 'TAKETOM', 'NOTSMAR', 'PHYSPRO']
        self.emotional_vars = ['EMOCTDW', 'EMOACCO', 'EMOCARE']
        self.social_health_vars = ['INTERFR', 'SOCIAL']
        self.cognitive_vars = ['TOTIDE1', 'TOTIDE2']
        self.symptom_vars = ['NITESWE', 'BOTCLDS', 'IRRITAB', 'MOODCHG']
        self.control_vars = ['STATUS', 'VISIT']
        self.socioeco_vars = ['INCOME', 'HOW_HAR', 'BCINCML', 'DEGREE']
        self.all_vars = self.social_vars + self.emotional_vars + self.social_health_vars + self.cognitive_vars + self.symptom_vars + self.control_vars
        self.output_dir = get_output_dir('3_social_model', 'overall')
        os.makedirs(self.output_dir, exist_ok=True)

    def preprocess_data(self):
        """Preprocess the data for analysis."""
        # Print initial data info for debugging
        
        # Coerce variables to numeric
        for col in self.all_vars:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        # For VISIT specifically, make sure we extract the number correctly
        if 'VISIT' in self.data.columns:
            if self.data['VISIT'].dtype == 'object':
                # Try to extract visit number if it's in format like 'VISIT_01'
                self.data['VISIT'] = self.data['VISIT'].str.extract('(\d+)').astype(float)
            elif self.data['VISIT'].dtype in ['int64', 'float64']:
                # Already numeric, no need to modify
                pass
            else:
                print(f"Warning: Unexpected VISIT type: {self.data['VISIT'].dtype}")

        # Drop rows with missing data
        self.data = self.data.dropna(subset=self.all_vars)
        
        # Ensure SWANID is treated as categorical
        self.data['SWANID'] = self.data['SWANID'].astype('category')
        
        # Sort data by SWANID and VISIT
        self.data = self.data.sort_values(['SWANID', 'VISIT'])

    def create_binary_groups(self, series):
        """Create binary groups (High/Low) handling duplicate values."""
        median = series.median()
        return pd.Series(np.where(series > median, 'High', 'Low'), index=series.index)

    def create_correlation_heatmap(self):
        """Create and save correlation heatmap for all variables."""
        plt.figure(figsize=(15, 12))
        correlation_matrix = self.data[self.all_vars].corr()
        
        # Create heatmap
        sns.heatmap(correlation_matrix, 
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   fmt='.2f',
                   square=True)
        
        plt.title('Correlation Heatmap of All Variables')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'correlation_heatmap.png'))
        plt.close()

    def create_interaction_plots(self, outcome_var, predictor_vars):
        """Create interaction plots between outcome and predictor variables."""
        n_predictors = len(predictor_vars)
        n_cols = 3
        n_rows = (n_predictors + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten()
        
        for i, predictor in enumerate(predictor_vars):
            # Create scatter plot with individual lines for each subject
            sns.lmplot(data=self.data, 
                      x=predictor, 
                      y=outcome_var,
                      hue='SWANID',
                      fit_reg=False,
                      scatter_kws={'alpha':0.3},
                      legend=False)
            
            # Add overall regression line
            plt.gca().axline(
                (self.data[predictor].mean(), 
                 self.data[outcome_var].mean()),
                slope=np.polyfit(self.data[predictor], 
                               self.data[outcome_var], 1)[0],
                color='red',
                label='Population trend'
            )
            
            # Calculate correlation coefficient
            corr, p_value = stats.pearsonr(self.data[predictor], self.data[outcome_var])
            plt.title(f'{predictor} vs {outcome_var}\nr={corr:.2f}, p={p_value:.3f}')
            
            plt.savefig(os.path.join(self.output_dir, f'{outcome_var}_{predictor}_interaction.png'))
            plt.close()
        
    def create_boxplots(self, outcome_var):
        """Create boxplots for categorical variables."""
        categorical_vars = ['STATUS']  # Add other categorical variables if needed
        
        fig, axes = plt.subplots(1, len(categorical_vars), figsize=(6*len(categorical_vars), 6))
        if len(categorical_vars) == 1:
            axes = [axes]
        
        for ax, cat_var in zip(axes, categorical_vars):
            # Create nested boxplot showing both overall and within-subject variation
            sns.boxplot(data=self.data, x=cat_var, y=outcome_var, ax=ax, color='lightgray')
            sns.stripplot(data=self.data, x=cat_var, y=outcome_var, ax=ax, 
                         hue='SWANID', size=4, alpha=0.3, jitter=0.2, legend=False)
            ax.set_title(f'{outcome_var} by {cat_var}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{outcome_var}_boxplots.png'))
        plt.close()

    def create_longitudinal_plots(self, outcome_var, predictor_vars):
        """Create combined longitudinal visualizations."""
        # Sort data by visit to ensure correct line plotting
        self.data = self.data.sort_values(['SWANID', 'VISIT'])
        
        # Create figure with 2x2 subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Basic trajectory plot (top left)
        ax1 = plt.subplot(221)
        # Individual trajectories
        for subject in self.data['SWANID'].unique():
            subject_data = self.data[self.data['SWANID'] == subject].sort_values('VISIT')
            ax1.plot(subject_data['VISIT'], subject_data[outcome_var], 
                    'gray', alpha=0.1, linewidth=0.5)
        
        # Mean trajectory with confidence interval
        mean_data = (self.data.groupby('VISIT')[outcome_var]
                    .agg(['mean', 'std'])
                    .reset_index())
        ax1.plot(mean_data['VISIT'], mean_data['mean'], 
                'r-', linewidth=2, label='Mean trajectory')
        ax1.fill_between(mean_data['VISIT'], 
                        mean_data['mean'] - mean_data['std'],
                        mean_data['mean'] + mean_data['std'],
                        color='red', alpha=0.2)
        
        ax1.set_title(f'{outcome_var} Individual Trajectories and Mean')
        ax1.set_xlabel('Visit')
        ax1.set_ylabel(outcome_var)
        ax1.set_xticks(range(1, 11))
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Status-stratified plot (top right)
        ax2 = plt.subplot(222)
        for status in sorted(self.data['STATUS'].unique()):
            status_data = self.data[self.data['STATUS'] == status]
            means = status_data.groupby('VISIT')[outcome_var].agg(['mean', 'std']).reset_index()
            ax2.plot(means['VISIT'], means['mean'], 
                    linewidth=2, label=f'Status {status}')
            ax2.fill_between(means['VISIT'],
                            means['mean'] - means['std'],
                            means['mean'] + means['std'],
                            alpha=0.1)
        ax2.set_title(f'{outcome_var} by Status')
        ax2.set_xlabel('Visit')
        ax2.set_ylabel(outcome_var)
        ax2.set_xticks(range(1, 11))
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Symptom effects (bottom left)
        ax3 = plt.subplot(223)
        symptom_vars = [var for var in predictor_vars if var in self.symptom_vars]
        if symptom_vars:
            for i, symptom in enumerate(symptom_vars[:3]):  # Plot first 3 symptoms
                # Create binary groups based on median
                median = self.data[symptom].median()
                self.data[f'temp_group_{i}'] = np.where(self.data[symptom] > median, 'High', 'Low')
                
                # Calculate means for each group
                for group in ['Low', 'High']:
                    group_data = self.data[self.data[f'temp_group_{i}'] == group]
                    means = group_data.groupby('VISIT')[outcome_var].mean()
                    ax3.plot(means.index, means.values,
                            label=f'{symptom} {group}',
                            linestyle='--' if group == 'Low' else '-')
                
                # Clean up temporary group column
                self.data.drop(f'temp_group_{i}', axis=1, inplace=True)
        
        ax3.set_title(f'{outcome_var} by Symptom Levels')
        ax3.set_xlabel('Visit')
        ax3.set_ylabel(outcome_var)
        ax3.set_xticks(range(1, 11))
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # 4. Social variable effects (bottom right)
        ax4 = plt.subplot(224)
        social_vars = [var for var in predictor_vars if var in self.social_vars]
        if social_vars:
            for i, social in enumerate(social_vars[:3]):  # Plot first 3 social variables
                # Create binary groups based on median
                median = self.data[social].median()
                self.data[f'temp_group_{i}'] = np.where(self.data[social] > median, 'High', 'Low')
                
                # Calculate means for each group
                for group in ['Low', 'High']:
                    group_data = self.data[self.data[f'temp_group_{i}'] == group]
                    means = group_data.groupby('VISIT')[outcome_var].mean()
                    ax4.plot(means.index, means.values,
                            label=f'{social} {group}',
                            linestyle='--' if group == 'Low' else '-')
                
                # Clean up temporary group column
                self.data.drop(f'temp_group_{i}', axis=1, inplace=True)
        
        ax4.set_title(f'{outcome_var} by Social Variable Levels')
        ax4.set_xlabel('Visit')
        ax4.set_ylabel(outcome_var)
        ax4.set_xticks(range(1, 11))
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{outcome_var}_combined_longitudinal.png'),
                    bbox_inches='tight')
        plt.close()

    def fit_mixed_model(self, outcome_var, predictors):
        """Fit mixed effects model with random intercept for each subject."""
        # Prepare the data
        model_data = self.data.copy()
        
        # Create design matrix X (fixed effects)
        X = pd.concat([model_data[predictors]], axis=1)
        X = sm.add_constant(X)
        
        # Fit mixed effects model
        model = MixedLM(model_data[outcome_var], 
                       X,
                       groups=model_data['SWANID'])
        
        try:
            # First try fitting with random intercept and slope
            result = model.fit()
        except:
            try:
                # If that fails, try just random intercept
                result = model.fit(reml=True)
            except:
                print(f"Warning: Model fitting failed for {outcome_var}")
                return None
                
        return result

    def run_complete_analysis(self):
        print("Preprocessing data...")
        self.preprocess_data()
        
        # Set up output capture
        output_capture = OutputCapture(self.output_dir)
        sys.stdout = output_capture

        # Print data info for debugging
        print("\nData information after preprocessing:")
        print(f"Total number of observations: {len(self.data)}")
        print("\nSummary of key variables:")
        print(self.data[['VISIT', 'STATUS'] + self.cognitive_vars].describe())
        
        try:
            # Create overall correlation heatmap
            print("Creating correlation heatmap...")
            self.create_correlation_heatmap()
            
            # Cognitive outcomes
            for outcome in self.cognitive_vars:
                print(f"\nAnalyzing {outcome}...")
                predictors = self.social_vars + self.symptom_vars + self.control_vars
                # Create visualizations
                self.create_longitudinal_plots(outcome, predictors)
                self.create_interaction_plots(outcome, self.social_vars + self.symptom_vars)
                self.create_boxplots(outcome)
                
                # Fit and print model
                model = self.fit_mixed_model(outcome, predictors)
                if model is not None:
                    print(model.summary())
                    
                    # Print random effects summary
                    print("\nRandom Effects Summary:")
                    print(model.random_effects)

            # Emotional health outcomes
            for outcome in self.emotional_vars:
                print(f"\nAnalyzing {outcome}...")
                # Create visualizations
                self.create_interaction_plots(outcome, self.social_vars + self.cognitive_vars + self.symptom_vars)
                self.create_boxplots(outcome)
                
                # Fit and print model
                model = self.fit_mixed_model(outcome, self.social_vars + self.cognitive_vars + self.symptom_vars + self.control_vars)
                if model is not None:
                    print(model.summary())
                    print("\nRandom Effects Summary:")
                    print(model.random_effects)

            # Social health outcomes
            for outcome in self.social_health_vars:
                print(f"\nAnalyzing {outcome}...")
                # Create visualizations
                self.create_interaction_plots(outcome, self.social_vars + self.cognitive_vars + self.emotional_vars + self.symptom_vars)
                self.create_boxplots(outcome)
                
                # Fit and print model
                model = self.fit_mixed_model(outcome, self.social_vars + self.cognitive_vars + self.emotional_vars + self.symptom_vars + self.control_vars)
                if model is not None:
                    print(model.summary())

            # Socioeconomic outcomes
            for outcome in self.socioeco_vars:
                print(f"\nAnalyzing {outcome}...")
                # Create visualizations
                self.create_interaction_plots(outcome, self.social_vars + self.cognitive_vars + self.emotional_vars + self.symptom_vars + self.social_health_vars)
                self.create_boxplots(outcome)
                
                # Fit and print model
                model = self.fit_mixed_model(outcome, self.social_vars + self.cognitive_vars + self.emotional_vars + self.symptom_vars + self.social_health_vars + self.control_vars)
                if model is not None:
                    print(model.summary())

        finally:
            sys.stdout = output_capture.terminal
            output_capture.close()

if __name__ == "__main__":
    file_path = "processed_combined_data.csv"
    analysis = MenopauseSocialCognitionModel(file_path)
    analysis.run_complete_analysis()