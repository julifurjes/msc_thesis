import pandas as pd
import numpy as np
import semopy
import os

class SEMAnalyzer:
    def __init__(self, model, data, params):
        """
        Initialize analyzer with fitted model, data, and parameters
        
        Args:
            model: Fitted semopy Model object
            data: DataFrame with observed variables
            params: DataFrame with parameter estimates
        """
        self.model = model
        self.data = data
        self.params = params
        
    def get_fit_indices(self):
        """Get model fit statistics using semopy's calc_stats function"""
        stats = semopy.calc_stats(self.model)
        return stats
    
    def format_fit_indices(self):
        """Format fit indices into a readable string"""
        stats = self.get_fit_indices()
        return stats.T
    
    def create_correlation_heatmaps(self):
        """Create correlation heatmaps for observed and latent variables"""
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        # Get variables used in the model
        model_vars = set()
        inspect_df = self.model.inspect()
        
        # Collect all observed variables from the model specification
        for _, row in inspect_df.iterrows():
            if row['op'] in ['=~', '~']:  # Measurement and structural relations
                if isinstance(row['lval'], str) and row['lval'] in self.data.columns:
                    model_vars.add(row['lval'])
                if isinstance(row['rval'], str) and row['rval'] in self.data.columns:
                    model_vars.add(row['rval'])
        
        # Remove any control variables that might be present
        model_vars = model_vars - {'STATUS', 'LANGCOG', 'SWANID'}  # Remove control variables and grouping var
        
        # Convert to list and sort for consistent ordering
        observed_vars = sorted(list(model_vars))
        
        if not observed_vars:
            print("Warning: No observed variables found in model specification")
            return None
            
        # Calculate correlations
        observed_corr = self.data[observed_vars].corr()
        
        print("Observed Variables Correlation Matrix:")
        print(observed_corr)
        
        try:
            # Get latent variable scores
            latent_scores = self.model.predict()
            latent_corr = latent_scores.corr()
        except Exception as e:
            print(f"Warning: Could not calculate latent variable correlations: {str(e)}")
            latent_corr = None
        
        # Determine number of plots needed
        n_plots = 2 if latent_corr is not None else 1
        fig, axes = plt.subplots(1, n_plots, figsize=(10 * n_plots, 8))
        
        if n_plots == 1:
            axes = [axes]
        
        # Observed variables heatmap
        mask1 = np.triu(np.ones_like(observed_corr), k=1)
        sns.heatmap(observed_corr, 
                   ax=axes[0],
                   cmap='RdBu_r',
                   vmin=-1,
                   vmax=1,
                   center=0,
                   annot=True,
                   fmt='.2f',
                   square=True,
                   mask=mask1)
        axes[0].set_title('Observed Variables Correlation Matrix', pad=20)
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
        axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0)
        
        # Latent variables heatmap (if available)
        if latent_corr is not None:
            mask2 = np.triu(np.ones_like(latent_corr), k=1)
            sns.heatmap(latent_corr,
                       ax=axes[1],
                       cmap='RdBu_r',
                       vmin=-1,
                       vmax=1,
                       center=0,
                       annot=True,
                       fmt='.2f',
                       square=True,
                       mask=mask2)
            axes[1].set_title('Latent Variables Correlation Matrix', pad=20)
            axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
            axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        return fig

class SEMVisualizer:
    def __init__(self, params_df):
        """
        Initialize visualizer with parameter estimates dataframe
        """
        self.params = params_df
        self.std_col = next(col for col in params_df.columns if "std" in col.lower())
        
    def create_mermaid_diagram(self, outcome, predictors):
        """Create a Mermaid diagram for a specific outcome and its predictors"""
        # Start Mermaid diagram
        mermaid = ["graph LR"]
        
        # Add nodes for predictors
        mermaid.append("    %% Predictor latent variables")
        for pred in predictors:
            display_name = self._format_display_name(pred)
            mermaid.append(f"    {pred}(({display_name}))")
        
        # Add outcome node
        mermaid.append("\n    %% Outcome latent variable")
        outcome_display = self._format_display_name(outcome)
        mermaid.append(f"    {outcome}(({outcome_display}))")
        
        # Add relationships
        mermaid.append("\n    %% Structural Model: Relationships among latent variables")
        for pred in predictors:
            coeff = self._get_coefficient(outcome, pred)
            if coeff is not None:
                # Use different line styles based on coefficient strength and sign
                if abs(coeff) < 0.1:
                    line_style = "-."
                    arrow = ".-> "
                else:
                    line_style = "--"
                    arrow = "--> "
                
                # Format coefficient with sign
                coeff_str = f"{coeff:+.2f}".replace("+", "+")
                mermaid.append(f'    {pred} {line_style} "{coeff_str}" {arrow}{outcome}')
        
        # Add style definitions
        mermaid.extend([
            "\n    %% Style definitions",
            "    classDef predictor fill:#cde,stroke:#333,stroke-width:2px;",
            "    classDef outcome fill:#fde,stroke:#333,stroke-width:2px;",
            f"    class {','.join(predictors)} predictor;",
            f"    class {outcome} outcome;"
        ])
        
        return "\n".join(mermaid)
    
    def create_html_visualization(self, mermaid_diagram):
        """Create an HTML file that renders the Mermaid diagram"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SEM Path Diagram</title>
            <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
            <script>
                mermaid.initialize({{
                    startOnLoad: true,
                    theme: 'default',
                    flowchart: {{
                        curve: 'basis',
                        nodeSpacing: 50,
                        rankSpacing: 50,
                        padding: 8
                    }}
                }});
            </script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .diagram-container {{
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .mermaid {{
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 400px;
                }}
            </style>
        </head>
        <body>
            <div class="diagram-container">
                <div class="mermaid">
                    {mermaid_diagram}
                </div>
            </div>
        </body>
        </html>
        """
        return html_content
    
    def visualize_all_outcomes(self):
        """Create Mermaid diagrams for all main outcomes"""
        outcome_specs = {
            "cognitive": ["social", "emotional", "social_health"],
            "emotional": ["social", "symptoms", "cognitive"],
            "social_health": ["social", "emotional", "symptoms", "cognitive"]
        }
        
        diagrams = {}
        for outcome, predictors in outcome_specs.items():
            diagrams[outcome] = self.create_mermaid_diagram(outcome, predictors)
            
        return diagrams
    
    def save_visualizations(self, output_dir):
        """Save diagrams as both text and HTML files"""
        os.makedirs(output_dir, exist_ok=True)
        
        diagrams = self.visualize_all_outcomes()
        for outcome, diagram in diagrams.items():
            # Save text version
            txt_path = os.path.join(output_dir, f"mermaid_{outcome}.txt")
            with open(txt_path, 'w') as f:
                f.write(diagram)
                
            # Save HTML version
            html_path = os.path.join(output_dir, f"mermaid_{outcome}.html")
            html_content = self.create_html_visualization(diagram)
            with open(html_path, 'w') as f:
                f.write(html_content)
                
            print(f"Saved diagram for {outcome} to:")
            print(f"  - Text: {txt_path}")
            print(f"  - HTML: {html_path}")
    
    def _format_display_name(self, var_name):
        """Format variable name for display"""
        if "_" in var_name:
            formatted = " ".join(word.capitalize() for word in var_name.split("_"))
            return formatted
        else:
            return var_name.capitalize()
    
    def _get_coefficient(self, outcome, predictor):
        """Get standardized coefficient for relationship"""
        row = self.params[
            (self.params["lval"] == outcome) & 
            (self.params["op"] == "~") & 
            (self.params["rval"] == predictor)
        ]
        if not row.empty:
            return row[self.std_col].values[0]
        return None