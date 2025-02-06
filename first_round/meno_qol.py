import pandas as pd
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt

class MenopauseQualityOfLifeModel:
    def __init__(self, data_path, target, predictors, status_col):
        """
        Initialize the model with data and variables.

        :param data_path: Path to the dataset (CSV file).
        :param target: Dependent variable (e.g., quality of life).
        :param predictors: List of independent variables (e.g., physical and mental health factors).
        :param status_col: Column representing menopausal status.
        """
        self.data_path = data_path
        self.target = target
        self.predictors = predictors
        self.status_col = status_col

    def load_and_prepare_data(self):
        """
        Load and preprocess the dataset.
        """
        # Load dataset
        self.data = pd.read_csv(self.data_path, low_memory=False)

        # Replace empty strings or spaces with NaN
        self.data.replace(r'^\s*$', pd.NA, regex=True, inplace=True)

        # Convert necessary columns to numeric
        cols_to_convert = [self.target] + self.predictors + [self.status_col]
        for col in cols_to_convert:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

        # Drop rows with missing values
        self.data = self.data.dropna(subset=cols_to_convert)

    def fit_model(self):
        """
        Fit a regression model to predict quality of life based on menopausal stage and health factors.
        """
        # Add interaction terms
        self.data['status_physical_interaction'] = self.data[self.status_col] * self.data['BODYPAI']
        self.data['status_mental_interaction'] = self.data[self.status_col] * self.data['DEPRESS']

        # Build formula for regression
        predictors_formula = " + ".join(self.predictors + ['status_physical_interaction', 'status_mental_interaction'])
        formula = f"{self.target} ~ {self.status_col} + {predictors_formula}"

        # Fit the model
        self.model = smf.ols(formula=formula, data=self.data).fit()

        # Print regression summary
        print(self.model.summary())
        return self.model

    def visualize_effects(self):
        """
        Visualize the effects of menopausal status on quality of life.
        """
        sns.lmplot(
            data=self.data,
            x=self.status_col,
            y=self.target,
            hue='BODYPAI',
            palette="coolwarm",
            height=5,
            aspect=1.2,
        )
        plt.title("Effect of Menopausal Status and Physical Health on Quality of Life")
        plt.show()

        sns.lmplot(
            data=self.data,
            x=self.status_col,
            y=self.target,
            hue='DEPRESS',
            palette="coolwarm",
            height=5,
            aspect=1.2,
        )
        plt.title("Effect of Menopausal Status and Mental Health on Quality of Life")
        plt.show()


if __name__ == "__main__":
    # Define the dataset and variables
    data_path = "combined_data.csv"
    target = "QLTYLIF"  # Quality of life
    predictors = ["BODYPAI", "OVERHLT", "DEPRESS", "FEARFUL"]  # Health factors
    status_col = "STATUS"  # Menopausal stage

    # Initialize the model
    model = MenopauseQualityOfLifeModel(data_path, target, predictors, status_col)

    # Load and prepare the data
    model.load_and_prepare_data()

    # Fit the regression model
    fitted_model = model.fit_model()

    # Visualize the effects
    model.visualize_effects()