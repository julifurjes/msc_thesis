import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt

class MenopauseMentalHealthAnalysis:
    def __init__(self, data_path, features, target, status_col, income_col, marital_col):
        """
        Initialize the analysis class with data and required columns.

        :param data_path: Path to the dataset (CSV file).
        :param features: List of feature column names.
        :param target: Target mental health variable (e.g., depression score).
        :param status_col: Column representing menopausal status.
        :param income_col: Column representing income level.
        :param marital_col: Column representing marital status.
        """
        self.data_path = data_path
        self.features = features
        self.target = target
        self.status_col = status_col
        self.income_col = income_col
        self.marital_col = marital_col

    def load_and_prepare_data(self):
        """
        Load and preprocess the dataset.
        """
        # Load dataset
        self.data = pd.read_csv(self.data_path, low_memory=False)

        # Replace empty strings or spaces with NaN
        self.data.replace(r'^\s*$', np.nan, regex=True, inplace=True)

        # Ensure features and target are numeric
        for col in self.features + [self.target, self.status_col, self.income_col, self.marital_col]:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

        # Drop rows with missing values
        self.data = self.data.dropna(subset=self.features + [self.target, self.status_col, self.income_col, self.marital_col])

        # Map marital status to binary (e.g., 1 = Married, 0 = Not Married)
        self.data[self.marital_col] = self.data[self.marital_col].apply(lambda x: 1 if x == 1 else 0)

    def interaction_analysis(self):
        """
        Perform interaction analysis to test the hypothesis.
        """
        # Add interaction terms to the data
        self.data['income_status_interaction'] = self.data[self.income_col] * self.data[self.status_col]
        self.data['marital_status_interaction'] = self.data[self.marital_col] * self.data[self.status_col]

        # Build a regression model with interaction terms
        formula = f"{self.target} ~ {self.status_col} + {self.income_col} + {self.marital_col} + " \
                  f"income_status_interaction + marital_status_interaction"
        model = smf.ols(formula=formula, data=self.data).fit()

        # Print regression summary
        print(model.summary())

        return model

    def visualize_interactions(self, model):
        """
        Visualize interaction effects.
        """
        sns.lmplot(
            data=self.data,
            x=self.status_col,
            y=self.target,
            hue=self.marital_col,
            col=self.income_col,
            palette="viridis",
            height=4,
            aspect=1
        )
        plt.title("Interaction Effects of Menopausal Status and Socioeconomic Factors on Mental Health")
        plt.show()


# Example Usage
if __name__ == "__main__":
    # Define file path and columns
    data_path = "combined_data.csv"
    features = [
        "AGE", "BODYPAI", "OVERHLT", "PHYCTDW"  # Example additional features
    ]
    target = "DEPRESS"  # Mental health variable (e.g., depression score)
    status_col = "STATUS"  # Menopausal status
    income_col = "INCOME"  # Income level
    marital_col = "MARITAL"  # Marital status

    # Instantiate the class
    analysis = MenopauseMentalHealthAnalysis(
        data_path, features, target, status_col, income_col, marital_col
    )

    # Load and prepare the data
    analysis.load_and_prepare_data()

    # Perform interaction analysis
    model = analysis.interaction_analysis()

    # Visualize interactions
    analysis.visualize_interactions(model)

# Higher income is weakly associated with lower depression scores