import pandas as pd
import numpy as np
from pyampute.exploration.mcar_statistical_tests import MCARTest

class DemographicAnalyzer:
    def __init__(self, data_file, output_file="demographic_summary.csv"):
        """
        Initialize the DemographicAnalyzer object.

        :param data_file: Path to the dataset file.
        :param output_file: The name of the file to save the demographic summary.
        """
        self.data_file = data_file
        self.output_file = output_file
        self.data = None
        self.summary = {}

    def load_data(self):
        """
        Load the dataset into a pandas DataFrame.
        """
        try:
            self.data = pd.read_csv(self.data_file, low_memory=False)
            print(f"Data successfully loaded from {self.data_file}.")
        except Exception as e:
            raise FileNotFoundError(f"Error loading data file: {e}")

    def clean_age_column(self):
        """
        Clean the AGE column by converting it to numeric and handling non-numeric values.
        """
        age_column = 'AGE'
        if age_column not in self.data.columns:
            raise ValueError(f"Age column '{age_column}' not found in the dataset.")

        # Convert the AGE column to numeric, coercing errors to NaN
        self.data[age_column] = pd.to_numeric(self.data[age_column], errors='coerce')

        # Drop rows where AGE is NaN
        original_row_count = len(self.data)
        self.data.dropna(subset=[age_column], inplace=True)
        cleaned_row_count = len(self.data)

        print(f"Cleaned the AGE column. Removed {original_row_count - cleaned_row_count} rows with invalid age values.")

    def calculate_age_stats(self):
        """
        Calculate mean and standard deviation of the AGE column.
        """
        age_column = 'AGE'
        if age_column not in self.data.columns:
            raise ValueError(f"Age column '{age_column}' not found in the dataset.")

        mean_age = self.data[age_column].mean()
        sd_age = self.data[age_column].std()
        self.summary['Mean Age'] = mean_age
        self.summary['Standard Deviation of Age'] = sd_age

        print(f"Mean Age: {mean_age:.2f}")
        print(f"Standard Deviation of Age: {sd_age:.2f}")

    def missing_data_summary(self):
        """
        Generate a summary of missing data in the dataset.
        """
        target_columns = ['STATUS', 'TOTIDE1', 'TOTIDE2']
        missing_data = self.data[target_columns].isnull().sum()
        missing_data_percentage = missing_data / len(self.data) * 100
        rounded_percentage = missing_data_percentage.round(2)

        for col in target_columns:
            self.summary[f'{col} Missing (%)'] = rounded_percentage[col]

        print("Missing Data Summary:")
        print(missing_data)

    def mcar_test(self):
        # Extract only numeric columns (since categorical variables can cause issues)
        columns_to_check = self.data[['TOTIDE1', 'TOTIDE2']]

        mt = MCARTest(method = 'little')

        # Perform Little's MCAR test
        p_value = mt(columns_to_check)

        print(f"Little's MCAR test p-value: {p_value:.5f}")
        self.summary["Little's MCAR Test P-Value"] = p_value

        if p_value > 0.05:
            print("The data is likely MCAR (missing completely at random).")
        else:
            print("The data is NOT MCAR (probably MAR or MNAR).")

    def save_summary(self):
        """
        Save the demographic summary to a CSV file.
        """
        summary_df = pd.DataFrame.from_dict(self.summary, orient='index', columns=['Value'])
        summary_df.to_csv(self.output_file)
        print(f"Demographic summary saved to {self.output_file}")

    def main(self):
        """
        Execute the demographic analysis workflow.
        """
        self.load_data()
        self.clean_age_column()
        self.calculate_age_stats()
        self.missing_data_summary()
        self.mcar_test()
        self.save_summary()


if __name__ == "__main__":
    data_file = "processed_data.csv"
    analyzer = DemographicAnalyzer(data_file)
    analyzer.main()