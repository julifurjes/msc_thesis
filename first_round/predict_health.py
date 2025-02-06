import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

class MenopauseMentalHealthCorrelation:
    def __init__(self, data_path, features, target_columns, id_column, time_column, status_column):
        """
        Initialize the analysis class with data and required columns.

        :param data_path: Path to the dataset (CSV file).
        :param features: List of feature column names (e.g., demographic, health indicators).
        :param target_columns: List of mental health/quality of life metrics.
        :param id_column: Column name for subject IDs.
        :param time_column: Column name for time/visit indicator.
        :param status_column: Column name for menopausal status.
        """
        self.data_path = data_path
        self.features = features
        self.target_columns = target_columns
        self.id_column = id_column
        self.time_column = time_column
        self.status_column = status_column

    def load_and_prepare_data(self):
        """
        Load and preprocess the dataset.
        """
        # Load dataset
        self.data = pd.read_csv(self.data_path, low_memory=False)

        # Replace empty strings or spaces with NaN
        self.data.replace(r'^\s*$', np.nan, regex=True, inplace=True)

        # Ensure features and target columns are numeric
        for col in self.features + self.target_columns:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

        # Map menopause status to ordinal values
        self.data[self.status_column] = pd.to_numeric(self.data[self.status_column], errors='coerce')

        # Drop rows with missing values
        self.data = self.data.dropna(subset=self.features + self.target_columns + [self.status_column])

    def correlation_analysis(self):
        """
        Perform correlation analysis between menopause status and mental health/quality of life.
        """
        corr_matrix = self.data[[self.status_column] + self.target_columns].corr(method="spearman")
        print("Correlation Matrix:")
        print(corr_matrix)

        # Heatmap for visualization
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Between Menopause Status and Mental Health/Quality of Life")
        plt.show()

    def regression_analysis(self):
        """
        Perform regression analysis to model the effect of menopause status on mental health/quality of life.
        """
        results = {}
        for target in self.target_columns:
            X = self.data[[self.status_column] + self.features]
            y = self.data[target]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Linear Regression
            lin_reg = LinearRegression()
            lin_reg.fit(X_train, y_train)
            y_pred = lin_reg.predict(X_test)

            # Metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            results[target] = {"MSE": mse, "R2": r2}

            print(f"Regression for Target: {target}")
            print(f"Mean Squared Error: {mse:.4f}")
            print(f"R2 Score: {r2:.4f}\n")

            # Feature importance for Random Forest
            rf = RandomForestRegressor(random_state=42)
            rf.fit(X_train, y_train)
            feature_importances = pd.DataFrame({
                "Feature": X.columns,
                "Importance": rf.feature_importances_
            }).sort_values(by="Importance", ascending=False)

            print(f"Feature Importance for Target: {target}")
            print(feature_importances)
            print()

        return results

# Example Usage
if __name__ == "__main__":
    # Define file path and columns
    data_path = "combined_data.csv"
    features = [
        "AGE", "INCOME", "MARITAL",  # Demographics
        "BODYPAI", "PHYCTDW", "OVERHLT",  # Physical health indicators
    ]
    target_columns = [
        "DEPRESS", "QLTYLIF", "FEARFUL", "SLEEPQL"  # Mental health and quality of life metrics
    ]
    id_column = "SWANID"
    time_column = "VISIT"
    status_column = "STATUS"  # Menopausal status

    # Instantiate the class
    correlation_model = MenopauseMentalHealthCorrelation(
        data_path, features, target_columns, id_column, time_column, status_column
    )

    # Load and prepare the data
    correlation_model.load_and_prepare_data()

    # Perform correlation analysis
    correlation_model.correlation_analysis()

    # Perform regression analysis
    correlation_model.regression_analysis()

# NO STRONG CORRELATION