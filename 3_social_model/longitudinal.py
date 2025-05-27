import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import networkx as nx
import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.save_output import OutputCapture, get_output_dir

class MenopauseCognitionMixedModels:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path, low_memory=False)
        self.output_dir = get_output_dir('3_social_model', 'longitudinal')
        
        # Define variable groups
        self.social_support_vars = ['LISTEN', 'TAKETOM', 'HELPSIC', 'CONFIDE']
        self.emotional_struggle_vars = ['EMOCTDW', 'EMOACCO', 'EMOCARE']
        self.social_struggle_vars = ['INTERFR', 'SOCIAL']
        self.cognitive_vars = ['TOTIDE1', 'TOTIDE2']
        self.symptom_vars = ['HOTFLAS', 'NUMHOTF', 'BOTHOTF', 'NITESWE', 'NUMNITS', 'BOTNITS',
                            'COLDSWE', 'NUMCLDS', 'BOTCLDS', 'STIFF', 'IRRITAB', 'MOODCHG']
        self.control_vars = ['INCOME', 'DEGREE', 'VISIT', 'STATUS']
        self.socioeco_vars = ['INCOME', 'DEGREE', 'HOW_HAR', 'BCINCML']

        self.model_coefficients = {}
        self.model_pvalues = {}
        self.model_results = {}
        
    def preprocess_data(self):
        """Prepare data for mixed-effects model analysis"""
        # Convert all variables to numeric
        relevant_vars = (self.social_support_vars + self.emotional_struggle_vars + self.social_struggle_vars + 
                   self.cognitive_vars + self.symptom_vars + self.control_vars + 
                   self.socioeco_vars + ['SWANID'])
        
        for col in relevant_vars:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

        if 'INCOME' in self.data.columns:
            # Convert INCOME to dummy variables
            # Store original INCOME for visualization purposes before creating dummies
            income_original = self.data['INCOME'].copy()
            # Convert INCOME to dummy variables
            income_dummies = pd.get_dummies(self.data['INCOME'], prefix='INCOME', drop_first=True)
            self.data = pd.concat([self.data, income_dummies], axis=1)
            # Keep the original for visualizations
            self.data['INCOME_original'] = income_original
            
        if 'DEGREE' in self.data.columns:
            # Convert DEGREE to dummy variables  
            # Store original DEGREE for visualization purposes before creating dummies
            degree_original = self.data['DEGREE'].copy()
            # Convert DEGREE to dummy variables  
            degree_dummies = pd.get_dummies(self.data['DEGREE'], prefix='DEGREE', drop_first=True)
            self.data = pd.concat([self.data, degree_dummies], axis=1)
            # Keep the original for visualizations
            self.data['DEGREE_original'] = degree_original
            
        if 'STATUS' in self.data.columns:
            self.data['STATUS'] = pd.to_numeric(self.data['STATUS'], errors='coerce')
            # Keep only statuses of interest (surgical: 1, 8; natural: 2,3,4,5)
            self.data = self.data[self.data['STATUS'].isin([1, 2, 3, 4, 5, 8])]
            
            # Map status to more descriptive labels for natural statuses and mark surgical statuses
            status_map = {
                1: 'Surgical',
                2: 'Post-menopause',
                3: 'Late Peri',
                4: 'Early Peri',
                5: 'Pre-menopause',
                8: 'Surgical'
            }
            self.data['STATUS_Label'] = self.data['STATUS'].map(status_map)
            
            # Convert STATUS to dummy variables for modeling
            status_dummies = pd.get_dummies(self.data['STATUS'], prefix='STATUS', drop_first=True)
            self.data = pd.concat([self.data, status_dummies], axis=1)

        # Filter dataset to keep only relevant columns
        self.data = self.data[relevant_vars + ['STATUS_Label', 'INCOME_original', 'DEGREE_original']]
        
        # Sort by subject ID and visit
        self.data = self.data.sort_values(['SWANID', 'VISIT'])
        
        # Ensure SWANID is treated as a categorical variable
        self.data['SWANID'] = self.data['SWANID'].astype(str)

        # Transform variables to address skewness
        self.transform_variables()
        
        # Create composite scores for each construct by averaging the indicator variables
        # This simplifies the analysis compared to latent variable modeling
        self.data['social_support'] = self.data[self.social_support_vars].mean(axis=1)
        self.data['emotional_struggle'] = self.data[self.emotional_struggle_vars].mean(axis=1)
        self.data['social_struggle'] = self.data[self.social_struggle_vars].mean(axis=1)
        self.data['cognitive_function'] = self.data[self.cognitive_vars].mean(axis=1)
        self.data['symptom_severity'] = self.data[self.symptom_vars].mean(axis=1)
        self.data['socioeconomic_status'] = self.data[self.socioeco_vars].mean(axis=1)
        
        # Calculate Cronbach's alpha for each composite score to check reliability
        self._calculate_reliability()

        # Due to the reliability check, the socioeconomic status variable will be included
        # As separate variables, so we need to drop na values from the specific variables
        
        # Drop rows with missing values in the composite scores or key variables
        composite_vars = ['social_support', 'emotional_struggle', 'social_struggle', 
                         'cognitive_function', 'symptom_severity'] + self.control_vars
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
        
        # Emotional Struggle
        alpha_emotional = cronbach_alpha(self.data[self.emotional_struggle_vars].dropna())
        print(f"Emotional Struggle: {alpha_emotional:.3f}")
        
        # Social Struggle
        alpha_social_struggle = cronbach_alpha(self.data[self.social_struggle_vars].dropna())
        print(f"Social Struggle: {alpha_social_struggle:.3f}")
        
        # Cognitive Function
        alpha_cognitive = cronbach_alpha(self.data[self.cognitive_vars].dropna())
        print(f"Cognitive Function: {alpha_cognitive:.3f}")
        
        # Symptoms
        alpha_symptoms = cronbach_alpha(self.data[self.symptom_vars].dropna())
        print(f"Symptom Severity: {alpha_symptoms:.3f}")

        # Socioeconomic Status
        alpha_socioeco = cronbach_alpha(self.data[self.socioeco_vars].dropna())
        print(f"Socioeconomic Status: {alpha_socioeco:.3f}")

    def transform_variables(self):
        """Apply transformations to address non-normality based on skewness values."""
        
        # Variables needing log transformation (high positive skew > 3.0)
        log_transform_vars = ['NERVES', 'NUMHOTF', 'NUMNITS', 'COLDSWE', 'NUMCLDS', 'FEARFULA']
        
        # Variables needing sqrt transformation (moderate positive skew 1.0-3.0)
        sqrt_transform_vars = ['SAD', 'HOTFLAS', 'NITESWE', 'IRRITAB', 'MOODCHG', 
                            'EMOCTDW', 'EMOACCO', 'EMOCARE', 'INTERFR', 'SOCIAL']
        
        # Variables with negative skew (need reflection then transformation)
        neg_skew_vars = ['TOTIDE1', 'TOTIDE2', 'LISTEN', 'TAKETOM', 'CONFIDE']
        
        # Variables to keep as-is (minimal skew between -1.0 and 1.0)
        no_transform_vars = ['BOTHOTF', 'BOTNITS', 'BOTCLDS', 'STIFF', 'HELPSIC', 'STATUS', 'INCOME', 'DEGREE']
        
        # Initialize transformed variable lists
        self.transformed_vars = []
        
        # Apply log transformations
        for var in log_transform_vars:
            if var in self.data.columns:
                self.data[f"{var}_log"] = np.log1p(self.data[var])  # log(x+1) to handle zeros
                self.transformed_vars.append(f"{var}_log")
        
        # Apply sqrt transformations
        for var in sqrt_transform_vars:
            if var in self.data.columns:
                self.data[f"{var}_sqrt"] = np.sqrt(self.data[var])
                self.transformed_vars.append(f"{var}_sqrt")
        
        # Handle negatively skewed variables - reflect, then transform
        # For variables like cognitive scores where higher is better
        for var in neg_skew_vars:
            if var in self.data.columns:
                # Find the maximum value for reflection
                max_val = self.data[var].max()
                # Reflect: subtract from max+1 so all values are positive
                reflected = (max_val + 1) - self.data[var]
                # Log transform the reflected values
                self.data[f"{var}_refl_log"] = np.log1p(reflected)
                self.transformed_vars.append(f"{var}_refl_log")
        
        # Add untransformed variables
        for var in no_transform_vars:
            if var in self.data.columns:
                self.transformed_vars.append(var)
        
        # Create transformed variable groups for model building
        self.transform_social_support = [f"{var}_refl_log" if var in neg_skew_vars 
                                    else f"{var}_sqrt" if var in sqrt_transform_vars
                                    else var for var in self.social_support_vars]
        
        self.transform_emotional = [f"{var}_sqrt" if var in sqrt_transform_vars else var 
                                for var in self.emotional_struggle_vars]
        
        self.transform_social_struggle = [f"{var}_sqrt" if var in sqrt_transform_vars else var 
                                    for var in self.social_struggle_vars]
        
        self.transform_cognitive = [f"{var}_refl_log" if var in neg_skew_vars else var 
                                for var in self.cognitive_vars]
        
        # Print transformation summary
        print("\nVariable Transformation Summary:")
        print(f"Log transformed ({len(log_transform_vars)}): {', '.join(log_transform_vars)}")
        print(f"Square root transformed ({len(sqrt_transform_vars)}): {', '.join(sqrt_transform_vars)}")
        print(f"Reflected and log transformed ({len(neg_skew_vars)}): {', '.join(neg_skew_vars)}")
        print(f"No transformation needed ({len(no_transform_vars)}): {', '.join(no_transform_vars)}")
    
    def fit_mixed_models(self):
        """Fit linear mixed-effects models for each outcome variable"""
        results = {}
        
        # Define the outcomes to model
        outcomes = {
            'cognitive_function': 'Cognitive Function',
            'emotional_struggle': 'Emotional Struggle',
            'social_struggle': 'Social Struggle',
            'symptom_severity': 'Symptom Severity'
        }
        
        for outcome_var, outcome_name in outcomes.items():
            print(f"\n\n{'='*50}")
            print(f"Modeling {outcome_name}")
            print(f"{'='*50}")
            
            # Get available dummy variables
            income_cols = [col for col in self.data.columns if col.startswith('INCOME_')]
            degree_cols = [col for col in self.data.columns if col.startswith('DEGREE_')]
            status_cols = [col for col in self.data.columns if col.startswith('STATUS_')]
            
            # Create formula with dummy variables instead of categorical
            predictors = ['social_support', 'emotional_struggle', 'social_struggle', 'symptom_severity']
            predictors += income_cols + degree_cols + status_cols + ['VISIT']
            
            # Remove the outcome variable from predictors if it appears
            predictors = [p for p in predictors if p != outcome_var]
            
            formula = f"{outcome_var} ~ " + " + ".join(predictors)
            
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
                
                # Store results
                results[outcome_var] = model_result
                self.model_results[outcome_var] = model_result
                
                # Extract coefficients and p-values for visualization methods
                params = model_result.params.drop(['Intercept', 'Group Var'], errors='ignore')
                pvalues = model_result.pvalues.drop(['Intercept', 'Group Var'], errors='ignore')
                
                # Store coefficients with standardized naming for visualization methods
                for param_name, coef in params.items():
                    # Create standardized keys for the visualization methods
                    if param_name == 'social_support':
                        if outcome_var == 'cognitive_function':
                            self.model_coefficients['Social_Support_to_Cognitive'] = coef
                            self.model_pvalues['Social_Support_to_Cognitive'] = pvalues[param_name]
                        elif outcome_var == 'emotional_struggle':
                            self.model_coefficients['Social_Support_to_Emotional'] = coef
                            self.model_pvalues['Social_Support_to_Emotional'] = pvalues[param_name]
                        elif outcome_var == 'social_struggle':
                            self.model_coefficients['Social_Support_to_Social_Health'] = coef
                            self.model_pvalues['Social_Support_to_Social_Health'] = pvalues[param_name]
                        elif outcome_var == 'symptom_severity':
                            self.model_coefficients['Social_Support_to_Symptom'] = coef
                            self.model_pvalues['Social_Support_to_Symptom'] = pvalues[param_name]
                    
                    elif param_name == 'emotional_struggle':
                        if outcome_var == 'cognitive_function':
                            self.model_coefficients['Emotional_to_Cognitive'] = coef
                            self.model_pvalues['Emotional_to_Cognitive'] = pvalues[param_name]
                        elif outcome_var == 'social_struggle':
                            self.model_coefficients['Emotional_to_Social_Health'] = coef
                            self.model_pvalues['Emotional_to_Social_Health'] = pvalues[param_name]
                        elif outcome_var == 'symptom_severity':
                            self.model_coefficients['Emotional_to_Symptom'] = coef
                            self.model_pvalues['Emotional_to_Symptom'] = pvalues[param_name]
                    
                    elif param_name == 'social_struggle':
                        if outcome_var == 'cognitive_function':
                            self.model_coefficients['Social_Health_to_Cognitive'] = coef
                            self.model_pvalues['Social_Health_to_Cognitive'] = pvalues[param_name]
                        elif outcome_var == 'emotional_struggle':
                            self.model_coefficients['Social_Health_to_Emotional'] = coef
                            self.model_pvalues['Social_Health_to_Emotional'] = pvalues[param_name]
                        elif outcome_var == 'symptom_severity':
                            self.model_coefficients['Social_Health_to_Symptom'] = coef
                            self.model_pvalues['Social_Health_to_Symptom'] = pvalues[param_name]
                    
                    elif param_name == 'symptom_severity':
                        if outcome_var == 'cognitive_function':
                            self.model_coefficients['Symptom_to_Cognitive'] = coef
                            self.model_pvalues['Symptom_to_Cognitive'] = pvalues[param_name]
                        elif outcome_var == 'emotional_struggle':
                            self.model_coefficients['Symptom_to_Emotional'] = coef
                            self.model_pvalues['Symptom_to_Emotional'] = pvalues[param_name]
                        elif outcome_var == 'social_struggle':
                            self.model_coefficients['Symptom_to_Social_Health'] = coef
                            self.model_pvalues['Symptom_to_Social_Health'] = pvalues[param_name]
                
                print("\nModel Summary:")
                print(model_result.summary())
                
                # Calculate model fit statistics
                self._calculate_model_fit(model_result, outcome_var)
                
            except Exception as e:
                print(f"Error fitting model for {outcome_name}: {str(e)}")
        
        return results
    
    def _adjust_formula(self, formula, outcome_var):
        """Adjust formula to prevent including the outcome as a predictor"""
        for var in ['social_support', 'emotional_struggle', 'social_struggle', 'symptom_severity']:
            if var == outcome_var:
                formula = formula.replace(f"{var} + ", "")
                formula = formula.replace(f"+ {var}", "")

        # Remove INCOME and DEGREE for cognitive function models
        if outcome_var == 'cognitive_function':
            formula = formula.replace("+ INCOME + DEGREE", "")
            formula = formula.replace("INCOME + DEGREE + ", "")

        return formula
    
    def _calculate_model_fit(self, model_result, outcome_var):
        """Calculate and print model fit statistics"""
        # Get the actual values
        y_true = model_result.model.endog
        
        # Get the predicted values
        y_pred = model_result.fittedvalues
        
        # Calculate RMSE (Root Mean Squared Error)
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        
        # Calculate mean absolute error
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Print fit statistics
        print("\nModel Fit Statistics:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        
    def create_correlation_network(self):
        """Create correlation network diagram with edge thickness proportional to correlation"""
        # Define the composite variables for the network
        network_vars = ['social_support', 'emotional_struggle', 'social_struggle', 
                    'symptom_severity', 'cognitive_function']
        
        # Calculate correlation matrix
        corr_matrix = self.data[network_vars].corr()
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes
        node_labels = {
            'social_support': 'Social\nSupport',
            'emotional_struggle': 'Emotional\nStruggle', 
            'social_struggle': 'Social\nStruggle',
            'symptom_severity': 'Symptom\nSeverity',
            'cognitive_function': 'Cognitive\nFunction'
        }
        
        for var in network_vars:
            G.add_node(var, label=node_labels[var])
        
        # Add edges with correlation as weight
        for i, var1 in enumerate(network_vars):
            for j, var2 in enumerate(network_vars):
                if i < j:  # Only add each edge once
                    corr_val = corr_matrix.loc[var1, var2]
                    if abs(corr_val) > 0.1:  # Only show correlations > 0.1
                        G.add_edge(var1, var2, weight=abs(corr_val), correlation=corr_val)
        
        # Create the plot
        plt.figure(figsize=(12, 10))
        
        # Position nodes in a circle
        pos = nx.circular_layout(G)
        
        # Define green color scheme from your palette
        node_color = self.green_palette[2]  # Medium green for nodes
        text_color = self.green_palette[7]  # Darkest green for text
        
        # Draw nodes with green theme
        nx.draw_networkx_nodes(G, pos, 
                            node_color=node_color,
                            node_size=3000,
                            alpha=0.8,
                            edgecolors=self.green_palette[5],  # Darker green border
                            linewidths=2)
        
        # Prepare edge data
        edges = list(G.edges())
        
        # Define complementary warm colors that work with green theme
        warm_colors = {
            'strong': '#B8860B',    # Dark goldenrod - strong contrast but earthy
            'moderate': '#DAA520',  # Goldenrod - visible but harmonious  
            'weak': '#F4E4BC'       # Light wheat - subtle but distinguishable
        }
        
        # Calculate colors and widths for each edge
        edge_colors = []
        edge_widths = []
        
        for edge in edges:
            correlation = G[edge[0]][edge[1]]['correlation']
            weight = G[edge[0]][edge[1]]['weight']
            
            # Calculate edge width (ensure minimum visibility)
            width = max(weight * 12, 3)
            edge_widths.append(width)
            
            # Calculate edge color based on correlation sign and strength
            if correlation > 0:
                # Positive correlations use green
                if weight > 0.6:
                    edge_colors.append(self.green_palette[7])  # Darkest green
                elif weight > 0.4:
                    edge_colors.append(self.green_palette[6])  # Dark green
                elif weight > 0.2:
                    edge_colors.append(self.green_palette[4])  # Medium green
                else:
                    edge_colors.append(self.green_palette[3])  # Light-medium green
            else:
                # Negative correlations use warm colors
                if weight > 0.6:
                    edge_colors.append(warm_colors['strong'])   # Dark goldenrod
                elif weight > 0.4:
                    edge_colors.append(warm_colors['moderate']) # Goldenrod
                else:
                    edge_colors.append(warm_colors['weak'])     # Light wheat
        
        # Draw edges with calculated colors and widths
        nx.draw_networkx_edges(G, pos,
                            edgelist=edges,
                            width=edge_widths,
                            edge_color=edge_colors,
                            alpha=0.9)
        
        # Draw labels with dark green color
        labels = {node: data['label'] for node, data in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels, 
                            font_size=18, 
                            font_weight='bold',
                            font_color=text_color)
        
        # Add correlation values as edge labels
        edge_labels = {}
        for u, v in G.edges():
            correlation = G[u][v]['correlation']
            edge_labels[(u, v)] = f'{correlation:.2f}'
        
        nx.draw_networkx_edge_labels(G, pos, edge_labels, 
                                    font_size=18,
                                    font_color=text_color,
                                    bbox=dict(boxstyle='round,pad=0.2', 
                                            facecolor='white', 
                                            edgecolor=self.green_palette[4],
                                            alpha=0.8))
        
        plt.axis('off')
        plt.tight_layout()
        
        # Save the plot
        network_path = os.path.join(self.output_dir, 'correlation_network.png')
        plt.savefig(network_path, dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Correlation network saved to: {network_path}")
        
        # Debug: Print edge information
        print("Edge correlation info:")
        for edge in edges:
            corr = G[edge[0]][edge[1]]['correlation']
            print(f"{edge}: {corr:.3f} ({'positive' if corr > 0 else 'negative'})")

    def create_composites(self):
        """Create composite scores - wrapper for visualization methods"""
        return self.data.copy()

    def correlation_heatmap(self, title="Correlation Matrix of Key Variables"):
        """Generate a correlation heatmap for key variables with nice labels"""
        # Use actual composite variables from the model
        variables = ['social_support', 'emotional_struggle', 'social_struggle', 
                    'cognitive_function', 'symptom_severity', 'INCOME_original', 'DEGREE_original']
        
        # Create a mapping for nicer variable labels
        label_mapping = {
            'social_support': 'Social Support',
            'emotional_struggle': 'Emotional Struggle',
            'social_struggle': 'Social Struggle',
            'cognitive_function': 'Cognitive Function',
            'symptom_severity': 'Symptom Severity',
            'INCOME_original': 'Income Level',
            'DEGREE_original': 'Education Level'
        }
        
        # Filter to keep only variables that exist in the dataframe
        variables = [var for var in variables if var in self.data.columns]
        
        if not variables:
            print("No variables available for correlation heatmap")
            return None
        
        # Create correlation matrix
        corr = self.data[variables].corr()
        
        # Rename the index and columns with nicer labels
        nice_labels = [label_mapping.get(var, var) for var in variables]
        corr.index = nice_labels
        corr.columns = nice_labels
        
        # Set up the figure
        plt.figure(figsize=self.figsize)
        
        # Draw the heatmap with a color bar
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', 
                    cmap=self.green_palette,
                    vmin=-1, vmax=1, center=0, square=True, linewidths=.5,
                    xticklabels=True, yticklabels=True,
                    annot_kws={'size': 14, 'weight': 'bold'})
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right', fontsize=14)
        plt.yticks(rotation=0, fontsize=14)
        plt.tight_layout()
        
        # Save the figure
        output_path = os.path.join(self.output_dir, 'correlation_matrix.png')
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def menopausal_status_effects(self, title="Cognitive, Emotional, Social, and Symptom Measures by Menopausal Status"):
        """Generate bar plots showing differences by menopausal status"""
        # Check if STATUS variable exists
        if 'STATUS_Label' not in self.data.columns:
            print(self.data.columns)
            print("Cannot create menopausal status effects plot - STATUS_Label variable missing")
            return None
            
        # Define domains to include
        domains = ['cognitive_function', 'emotional_struggle', 'social_struggle', 'symptom_severity']
        domains = [d for d in domains if d in self.data.columns]
        
        if not domains:
            print("Cannot create menopausal status effects plot - no domain variables available")
            return None
            
        # Create a summary dataframe with mean values by menopausal status
        summary = self.data.groupby('STATUS_Label')[domains].mean().reset_index()
        
        # Melt the dataframe for easier plotting
        melted = pd.melt(summary, id_vars=['STATUS_Label'], 
                         value_vars=domains,
                         var_name='Domain', value_name='Score')
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        sns.barplot(x='STATUS_Label', y='Score', hue='Domain', data=melted)
        
        plt.title(title, fontsize=16)
        plt.xlabel('Menopausal Status', fontsize=14)
        plt.ylabel('Average Score (Standardized)', fontsize=14)
        plt.xticks(rotation=45)
        plt.legend(title='Domain', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Save the figure
        output_path = os.path.join(self.output_dir, 'menopausal_status_effects.png')
        plt.savefig(output_path, dpi=self.dpi)
        plt.close()
        
        return output_path
        
    def create_path_coefficient_forest(self, results):
        """Create separate forest plots for each outcome variable (Table XI visualization)"""
        # Create a 2x2 subplot grid for the four outcomes
        fig, axes = plt.subplots(4, 1, figsize=(12, 20))
        axes = axes.flatten()
        
        # Import the palette and select distinct colors (matching second plot style)
        green_palette = sns.color_palette("YlGn", n_colors=8)
        selected_greens = [green_palette[1], green_palette[3], green_palette[5], green_palette[7]]
        
        # Define color mapping for p-values using selected colors from YlGn palette
        def get_color(p_value):
            if p_value < 0.001:
                return selected_greens[3], '***'  # Most significant: darkest green
            elif p_value < 0.01:
                return selected_greens[2], '**'   # Very significant: medium-dark green
            elif p_value < 0.05:
                return selected_greens[1], '*'    # Significant: light-medium green
            else:
                return 'gray', ''                 # Not significant: gray
        
        # Define outcome order and nice names
        outcomes_order = ['cognitive_function', 'emotional_struggle', 'social_struggle', 'symptom_severity']
        outcome_names = {
            'cognitive_function': 'Cognitive Function',
            'emotional_struggle': 'Emotional Struggle', 
            'social_struggle': 'Social Struggle',
            'symptom_severity': 'Symptom Severity'
        }
        
        # Define predictor order and nice names
        predictor_order = ['social_support', 'emotional_struggle', 'social_struggle', 
                        'symptom_severity', 'INCOME', 'DEGREE', 'VISIT', 'STATUS']
        predictor_names = {
            'social_support': 'Social Support',
            'emotional_struggle': 'Emotional Struggle',
            'social_struggle': 'Social Struggle', 
            'symptom_severity': 'Symptom Severity',
            'INCOME': 'Income',
            'DEGREE': 'Education',
            'VISIT': 'Visit',
            'STATUS': 'Status'
        }
        
        for idx, outcome in enumerate(outcomes_order):
            if outcome not in results:
                continue
                
            ax = axes[idx]
            result = results[outcome]
            
            # Get coefficients and standard errors
            params = result.params.drop(['Intercept', 'Group Var'], errors='ignore')
            std_errors = result.bse.drop(['Intercept', 'Group Var'], errors='ignore')
            
            # Calculate confidence intervals
            ci_lower = params - 1.96 * std_errors
            ci_upper = params + 1.96 * std_errors
            
            # Calculate p-values
            t_values = params / std_errors
            p_values = [2 * (1 - stats.t.cdf(abs(t), result.df_resid)) for t in t_values]
            
            # Filter and order predictors
            plot_data = []
            for pred in predictor_order:
                if pred in params.index and pred != outcome.replace('_function', '').replace('_struggle', '').replace('_severity', ''):
                    p_val = p_values[list(params.index).index(pred)]
                    color, stars = get_color(p_val)
                    plot_data.append({
                        'predictor': predictor_names.get(pred, pred),
                        'coef': params[pred],
                        'ci_lower': ci_lower[pred],
                        'ci_upper': ci_upper[pred],
                        'p_value': p_val,
                        'color': color,
                        'stars': stars,
                        'std_error': std_errors[pred]
                    })
            
            if not plot_data:
                continue
                
            # Create DataFrame for plotting
            plot_df = pd.DataFrame(plot_data)
            
            # Sort by coefficient magnitude for better visualization (matching second plot)
            plot_df = plot_df.reindex(plot_df['coef'].abs().sort_values(ascending=False).index)
            plot_df = plot_df.reset_index(drop=True)
            
            # Plot
            y_pos = np.arange(len(plot_df))
            
            # Plot each point individually with its own color (matching second plot style)
            for i, row in plot_df.iterrows():
                ax.errorbar(
                    row['coef'], y_pos[i],
                    xerr=1.96 * row['std_error'],  # 95% CI
                    fmt='o',
                    color=row['color'],
                    capsize=5,
                    markersize=8,
                    elinewidth=2,
                    capthick=2
                )
            
            # Add vertical line at zero
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            
            # Calculate reasonable x-axis limits based on coefficients and errors
            all_values = []
            for _, row in plot_df.iterrows():
                all_values.extend([row['ci_lower'], row['ci_upper']])
            
            if all_values:
                x_min, x_max = min(all_values), max(all_values)
                x_range = x_max - x_min
                ax.set_xlim(x_min - 0.1 * x_range, x_max + 0.3 * x_range)  # Extra space on right for labels
            
            # Add text labels with coefficients and significance stars - aligned to the right
            for i, row in plot_df.iterrows():
                # Position label to the right of the error bar with some padding
                label_x = row['ci_upper'] + 0.02 * (x_range if 'x_range' in locals() else 0.1)
                
                ax.text(
                    label_x, y_pos[i],
                    f'{row["coef"]:.3f} {row["stars"]}',
                    verticalalignment='center',
                    horizontalalignment='left',
                    fontsize=16,
                    color=row['color'],
                    fontweight='bold' if row['p_value'] < 0.05 else 'normal',
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2', edgecolor='none')
                )
            
            # Formatting
            ax.set_yticks(y_pos)
            ax.set_yticklabels(plot_df['predictor'], fontsize=18)
            ax.set_title(f'Effects on {outcome_names[outcome]}', fontsize=18, fontweight='bold')
            
            # Add gridlines (matching second plot style)
            ax.grid(True, axis='x', linestyle=':', alpha=0.6)
        
        # Remove empty subplots
        for idx in range(len(outcomes_order), len(axes)):
            fig.delaxes(axes[idx])
        
        # Create custom legend for significance levels and colors (matching second plot)
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=selected_greens[3], label='p < 0.001 (***)', edgecolor='black'),
            Patch(facecolor=selected_greens[2], label='p < 0.01 (**)', edgecolor='black'),
            Patch(facecolor=selected_greens[1], label='p < 0.05 (*)', edgecolor='black'),
            Patch(facecolor='gray', label='p ≥ 0.05 (n.s.)', edgecolor='black')
        ]
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.88))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        
        # Save the plot
        forest_path = os.path.join(self.output_dir, 'path_coefficient_forest.png')
        plt.savefig(forest_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Path coefficient forest plot saved to: {forest_path}")
        
    def create_visualizations(self, results):
        """Generate all visualizations for model results"""
        print("\nCreating visualizations...")
        self.green_palette = sns.color_palette("YlGn", n_colors=8)
        self.figsize = (10, 8)
        self.dpi = 300
        
        # Create original coefficient plot
        self._create_coefficient_plot(results)
        
        # Create correlation network and path coefficient forest
        self.create_correlation_network()
        self.create_path_coefficient_forest(results)
        
        # Create integrated visualization plots (from first paste)
        print("Creating integrated visualization plots...")
        
        # Correlation heatmap
        heatmap_path = self.correlation_heatmap()
        if heatmap_path:
            print(f"Correlation heatmap saved to: {heatmap_path}")
        
        # Menopausal status effects
        status_path = self.menopausal_status_effects()
        if status_path:
            print(f"Menopausal status effects saved to: {status_path}")
        
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
        
        # Also create a table of coefficients
        coef_table = coef_df.pivot(index='Predictor', columns='Outcome', values='Coefficient')
        
        # Save table to CSV
        table_path = os.path.join(self.output_dir, 'coefficient_table.csv')
        coef_table.to_csv(table_path)
        
        print(f"Coefficient table saved to: {table_path}")
        
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
            desc_stats = self.data[['social_support', 'emotional_struggle', 'social_struggle', 
                                   'cognitive_function', 'symptom_severity', 'socioeconomic_status']].describe()
            print(desc_stats)
            
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