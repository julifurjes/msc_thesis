import pandas as pd
import numpy as np
import os

class DemographicAnalyzer:
    def __init__(self, data_file, cross_sectional_file, output_file="demographic_summary.csv"):
        """
        Initialize the DemographicAnalyzer object.

        :param data_file: Path to the dataset file.
        :param output_file: The name of the file to save the demographic summary.
        """
        self.data_file = data_file
        self.output_file = output_file
        self.data = None
        self.cross_data_file = cross_sectional_file
        self.summary = {}

    def load_data(self):
        """
        Load the dataset into a pandas DataFrame.
        """
        try:
            self.data = pd.read_csv(self.data_file, low_memory=False)
            print(f"Data successfully loaded from {self.data_file}.")
            self.cross_data = pd.read_csv(self.cross_data_file, sep='\t', low_memory=False)
            print(f"Cross-sectional data successfully loaded from {self.cross_data_file}.\n")
        except Exception as e:
            raise FileNotFoundError(f"Error loading data file: {e}")

    def analyze_cross_sectional_data(self):
        """
        Analyze the cross-sectional dataset for AGE, RACE, and DEGREE distributions.
        """
        target_columns = ['RACE', 'DEGREE']

        # Compute mean and standard deviation for AGE
        if 'AGE' in self.cross_data.columns:
            self.cross_data['AGE'] = pd.to_numeric(self.cross_data['AGE'], errors='coerce')
            mean_age = self.cross_data['AGE'].mean()
            sd_age = self.cross_data['AGE'].std()
            self.summary['Cross-Sectional Mean Age'] = mean_age
            self.summary['Cross-Sectional Standard Deviation of Age'] = sd_age

            print(f"Cross-Sectional Mean Age: {mean_age:.2f}")
            print(f"Cross-Sectional Standard Deviation of Age: {sd_age:.2f}\n")
        else:
            print("Column 'AGE' not found in the cross-sectional dataset.")

        # Compute distributions for categorical variables
        for col in target_columns:
            if col in self.cross_data.columns:
                value_counts = self.cross_data[col].value_counts(dropna=False)
                self.summary[f'{col} Distribution'] = value_counts.to_dict()
                print(f"{col} Distribution:")
                print(value_counts, "\n")
            else:
                print(f"Column '{col}' not found in the cross-sectional dataset.")

    def analyze_status_distribution(self):
        """
        Analyze the STATUS distribution from self.data.
        """
        if 'STATUS' in self.data.columns:
            status_counts = self.data['STATUS'].value_counts(dropna=False)
            self.summary['STATUS Distribution'] = status_counts.to_dict()
            print("STATUS Distribution:")
            print(status_counts, "\n")
        else:
            print("Column 'STATUS' not found in the longitudinal dataset.")

    def count_complete_participants(self):
        """
        Count how many participants (SWANID) have data for all 10 visits and print total subjects.
        Also, create a structured distribution of how many participants completed each number of visits.
        Additionally, calculate N, percentage retention, and age statistics for each visit (0-10).
        """
        if 'SWANID' not in self.data.columns or 'VISIT' not in self.data.columns:
            raise ValueError("Required columns 'SWANID' and 'VISIT' not found in the longitudinal dataset.")

        total_subjects = self.data['SWANID'].nunique()
        visit_counts = self.data.groupby('SWANID')['VISIT'].nunique()
        
        # Count how many participants completed each number of visits (from 10 to 1)
        visit_distribution = visit_counts.value_counts().sort_index(ascending=False)

        self.summary['Total Subjects'] = total_subjects
        self.summary['Visit Completion Distribution'] = visit_distribution.to_dict()

        print(f"Total number of unique subjects: {total_subjects}")
        print("Visit Completion Distribution:")

        for visits, count in visit_distribution.items():
            print(f"{visits} visits completed: {count} participants")

        # Calculate statistics for each visit
        visit_stats = []
        for visit in range(11):  # Visits 0-10
            visit_data = self.data[self.data['VISIT'] == visit]
            
            # Count participants in this visit
            n_participants = visit_data['SWANID'].nunique()
            
            # Calculate retention percentage
            retention_percentage = round((n_participants / total_subjects) * 100, 1)
            
            # Calculate age statistics if 'AGE' column exists
            age_mean = None
            if 'AGE' in self.data.columns:
                age_data = visit_data['AGE'].dropna()
                if len(age_data) > 0:
                    age_mean = round(age_data.mean(), 2)
            
            visit_stats.append({
                'Visit': visit,
                'N': n_participants,
                'Retention (%)': retention_percentage,
                'Age Mean': age_mean
            })
        
        # Create a DataFrame for better visualization
        visit_stats_df = pd.DataFrame(visit_stats)
        self.summary['Visit Statistics'] = visit_stats_df
        
        print("\nVisit Statistics:")
        print(visit_stats_df.to_string(index=False))
        
        return visit_stats_df

    def missing_data_summary(self):
        """
        Generate a summary of missing data in the dataset and print total rows.
        """
        target_columns = ['STATUS', 'TOTIDE1', 'TOTIDE2']
        missing_data = self.data[target_columns].isnull().sum()
        total_rows = len(self.data)

        for col in target_columns:
            self.summary[f'{col} Missing'] = missing_data[col]


        self.summary['Total Rows'] = total_rows
        print(f"Total number of rows in dataset: {total_rows}")
        print("Missing Data Summary:")
        print(missing_data, "\n")

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
        self.analyze_cross_sectional_data()
        self.count_complete_participants()
        self.missing_data_summary()
        self.save_summary()


if __name__ == "__main__":
    data_file = "processed_data.csv"
    cross_sectional_file = os.path.join("data", "swan_cross_sectional.tsv")
    analyzer = DemographicAnalyzer(data_file, cross_sectional_file)
    analyzer.main()