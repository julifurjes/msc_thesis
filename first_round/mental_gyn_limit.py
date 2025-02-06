import pandas as pd
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt

class EverydayLifeImpactModel:
    def __init__(self, data_path, predictors, dependents):
        """
        Initialize the model with data and variables.

        :param data_path: Path to the dataset (CSV file).
        :param predictors: List of independent variables (mental health, gynecological symptoms).
        :param dependents: List of dependent variables (physical, emotional, social functioning).
        """
        self.data_path = data_path
        self.predictors = predictors
        self.dependents = dependents

    def load_and_prepare_data(self):
        """
        Load and preprocess the dataset.
        """
        # Load dataset
        self.data = pd.read_csv(self.data_path, low_memory=False)

        # Replace empty strings or spaces with NaN
        self.data.replace(r'^\s*$', pd.NA, regex=True, inplace=True)

        # Convert necessary columns to numeric
        all_columns = self.predictors + self.dependents
        for col in all_columns:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

        # Drop rows with missing values
        self.data = self.data.dropna(subset=all_columns)

    def fit_models(self):
        """
        Fit regression models for each dependent variable and store the results.
        """
        self.models = {}
        for dependent in self.dependents:
            formula = f"{dependent} ~ {' + '.join(self.predictors)}"
            model = smf.ols(formula=formula, data=self.data).fit()
            self.models[dependent] = model

            # Print summary for each model
            print(f"\nRegression Results for {dependent}:\n")
            print(model.summary())

    def visualize_results(self):
        """
        Visualize the impact of predictors on each dependent variable.
        """
        for dependent in self.dependents:
            sns.lmplot(
                data=self.data,
                x="DEPRESS",  # Example predictor
                y=dependent,
                height=5,
                aspect=1.2,
                scatter_kws={"alpha": 0.5},
            )
            plt.title(f"Relationship between DEPRESS and {dependent}")
            plt.xlabel("Depression (DEPRESS)")
            plt.ylabel(f"{dependent}")
            plt.show()


if __name__ == "__main__":
    # Define the dataset and variables
    data_path = "combined_data.csv"
    predictors = [
        "DEPRESS", "BLUES", "CONTROL", "PILING", "RESTLES", "SLEEPQL", "QLTYLIF",  # Mental health
        "ENDO", "PELVCPN", "ABBLEED", "FIBRUTR", "STATUS",  # Gynecological symptoms
    ]
    dependents = [
        "PHYCTDW", "PHYACCO", "PHYLIMI", "PHYDFCL",  # Physical functioning
        "EMOCTDW", "EMOACCO",  # Emotional functioning
        "INTERFR",  # Social functioning
    ]

    # Initialize and run the model
    impact_model = EverydayLifeImpactModel(data_path, predictors, dependents)

    # Load and prepare the data
    impact_model.load_and_prepare_data()

    # Fit regression models
    impact_model.fit_models()

    # Visualize results
    impact_model.visualize_results()