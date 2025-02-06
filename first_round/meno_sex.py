import pandas as pd
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt

class SexualHealthQualityOfLifeModel:
    def __init__(self, data_path, target, predictors, control_variable):
        """
        Initialize the model with data and variables.

        :param data_path: Path to the dataset (CSV file).
        :param target: Dependent variable (e.g., quality of life).
        :param predictors: List of independent variables (e.g., sexual health scores).
        :param control_variable: Control variable (e.g., menopausal status).
        """
        self.data_path = data_path
        self.target = target
        self.predictors = predictors
        self.control_variable = control_variable

    def load_and_prepare_data(self):
        """
        Load and preprocess the dataset.
        """
        # Load dataset
        self.data = pd.read_csv(self.data_path, low_memory=False)

        # Replace empty strings or spaces with NaN
        self.data.replace(r'^\s*$', pd.NA, regex=True, inplace=True)

        # Convert necessary columns to numeric
        cols_to_convert = [self.target] + self.predictors + [self.control_variable]
        for col in cols_to_convert:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

        # Drop rows with missing values
        self.data = self.data.dropna(subset=cols_to_convert)

    def fit_model(self):
        """
        Fit a regression model to predict quality of life based on sexual health variables.
        """
        # Build formula for regression
        predictors_formula = " + ".join(self.predictors + [self.control_variable])
        formula = f"{self.target} ~ {predictors_formula}"

        # Fit the model
        self.model = smf.ols(formula=formula, data=self.data).fit()

        # Print summary
        print(self.model.summary())
        return self.model

    def visualize_relationships(self):
        """
        Visualize the relationships between sexual health variables and quality of life.
        """
        for predictor in self.predictors:
            sns.lmplot(
                data=self.data,
                x=predictor,
                y=self.target,
                hue=self.control_variable,
                palette="coolwarm",
                height=5,
                aspect=1.2,
            )
            plt.title(f"Relationship between {predictor} and {self.target}")
            plt.show()


if __name__ == "__main__":
    # Define the dataset and variables
    data_path = "combined_data.csv"
    target = "QLTYLIF"  # Quality of life
    predictors = ["IMPORSE", "DESIRSE", "ENGAGSE"]  # Sexual health variables
    control_variable = "STATUS"  # Menopausal status

    # Initialize the model
    model = SexualHealthQualityOfLifeModel(data_path, target, predictors, control_variable)

    # Load and prepare the data
    model.load_and_prepare_data()

    # Fit the regression model
    fitted_model = model.fit_model()

    # Visualize relationships
    model.visualize_relationships()