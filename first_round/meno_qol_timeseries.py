import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class TimeSeriesQualityOfLifeModel:
    def __init__(self, data_path, target, predictors, status_col, id_col, time_col):
        """
        Initialize the time-series analysis model.

        :param data_path: Path to the dataset (CSV file).
        :param target: Dependent variable (e.g., quality of life).
        :param predictors: List of time-varying predictors (e.g., physical and mental health).
        :param status_col: Column representing menopausal status.
        :param id_col: Column representing subject ID.
        :param time_col: Column representing time/visit indicator.
        """
        self.data_path = data_path
        self.target = target
        self.predictors = predictors
        self.status_col = status_col
        self.id_col = id_col
        self.time_col = time_col

    def load_and_prepare_data(self):
        """
        Load and preprocess the dataset for time-series analysis.
        """
        # Load dataset
        self.data = pd.read_csv(self.data_path,  low_memory=False)

        # Replace empty strings or spaces with NaN
        self.data.replace(r'^\s*$', pd.NA, regex=True, inplace=True)

        # Convert necessary columns to numeric
        cols_to_convert = [self.target] + self.predictors + [self.status_col, self.time_col]
        for col in cols_to_convert:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

        # Drop rows with missing values
        self.data = self.data.dropna(subset=cols_to_convert)

    def fit_fixed_effects_model(self):
        """
        Fit a fixed-effects model to analyze quality of life based on menopausal status, physical, and mental health.
        """
        # Simplify STATUS by grouping rare categories
        self.data[self.status_col] = self.data[self.status_col].replace(
            {6.0: 5.0, 7.0: 5.0, 8.0: 5.0}  # Combine rare categories into broader groups
        )

        # Create formula for fixed-effects regression
        predictors_formula = " + ".join(self.predictors)
        formula = f"{self.target} ~ {self.status_col} + {predictors_formula}"

        # Fit the fixed-effects model
        model = smf.ols(formula=formula, data=self.data).fit()

        # Print model summary
        print(model.summary())
        return model

    def visualize_trends(self):
        """
        Visualize the trends in quality of life over time stratified by menopausal status.
        """
        sns.lineplot(
            data=self.data,
            x=self.time_col,
            y=self.target,
            hue=self.status_col,
            ci="sd",
            palette="coolwarm",
        )
        plt.title("Trends in Quality of Life Over Time by Menopausal Status")
        plt.xlabel("Visit (Time)")
        plt.ylabel("Quality of Life")
        plt.legend(title="Menopausal Status")
        plt.show()


if __name__ == "__main__":
    # Define the dataset and variables
    data_path = "combined_data.csv"
    target = "QLTYLIF"  # Quality of life
    predictors = ["BODYPAI", "OVERHLT", "DEPRESS", "FEARFUL"]  # Physical and mental health factors
    status_col = "STATUS"  # Menopausal status
    id_col = "SWANID"  # Subject ID
    time_col = "VISIT"  # Time/visit indicator

    # Initialize the model
    model = TimeSeriesQualityOfLifeModel(data_path, target, predictors, status_col, id_col, time_col)

    # Load and prepare the data
    model.load_and_prepare_data()

    # Fit the mixed-effects model
    fitted_model = model.fit_fixed_effects_model()

    # Visualize the trends
    model.visualize_trends()