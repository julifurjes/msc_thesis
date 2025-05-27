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

    def analyze_demographics_of_actual_participants(self):
        """
        Analyze demographic data (RACE, DEGREE, AGE) only for the participants
        who actually appear in the longitudinal dataset.
        """
        # Get the unique SWANID values from the longitudinal dataset
        if 'SWANID' not in self.data.columns:
            raise ValueError("SWANID column not found in the longitudinal dataset.")
            
        actual_participants = self.data['SWANID'].unique()
        print(f"Total number of unique participants in longitudinal data: {len(actual_participants)}")
        
        # Filter the cross-sectional data to include only these participants
        if 'SWANID' not in self.cross_data.columns:
            raise ValueError("SWANID column not found in the cross-sectional dataset.")
            
        filtered_cross_data = self.cross_data[self.cross_data['SWANID'].isin(actual_participants)]
        print(f"Number of participants found in cross-sectional data: {len(filtered_cross_data)}")
        
        # Analyze RACE distribution
        if 'RACE' in filtered_cross_data.columns:
            race_counts = filtered_cross_data['RACE'].value_counts(dropna=False)
            race_percentages = (race_counts / len(filtered_cross_data) * 100).round(1)
            
            # Combine counts and percentages
            race_distribution = pd.DataFrame({
                'Count': race_counts,
                'Percentage': race_percentages
            })
            
            self.summary['RACE Distribution of Actual Participants'] = race_distribution.to_dict()
            print("\nRACE Distribution of Actual Participants:")
            print(race_distribution)
        else:
            print("Column 'RACE' not found in the cross-sectional dataset.")
            
        # Analyze DEGREE distribution
        if 'DEGREE' in filtered_cross_data.columns:
            degree_counts = filtered_cross_data['DEGREE'].value_counts(dropna=False)
            degree_percentages = (degree_counts / len(filtered_cross_data) * 100).round(1)
            
            # Combine counts and percentages
            degree_distribution = pd.DataFrame({
                'Count': degree_counts,
                'Percentage': degree_percentages
            })
            
            self.summary['DEGREE Distribution of Actual Participants'] = degree_distribution.to_dict()
            print("\nDEGREE Distribution of Actual Participants:")
            print(degree_distribution)
        else:
            print("Column 'DEGREE' not found in the cross-sectional dataset.")
            
        # Analyze AGE distribution
        if 'AGE' in filtered_cross_data.columns:
            filtered_cross_data['AGE'] = pd.to_numeric(filtered_cross_data['AGE'], errors='coerce')
            mean_age = filtered_cross_data['AGE'].mean()
            sd_age = filtered_cross_data['AGE'].std()
            self.summary['Mean Age of Actual Participants'] = mean_age
            self.summary['Standard Deviation of Age of Actual Participants'] = sd_age

            print(f"\nMean Age of Actual Participants: {mean_age:.2f}")
            print(f"Standard Deviation of Age of Actual Participants: {sd_age:.2f}")
        else:
            print("Column 'AGE' not found in the cross-sectional dataset.")

    def analyze_status_distribution(self):
        """
        Analyze the STATUS distribution from self.data.
        """
        if 'STATUS' in self.data.columns:
            status_counts = self.data['STATUS'].value_counts(dropna=False)
            self.summary['STATUS Distribution'] = status_counts.to_dict()
            print("\nSTATUS Distribution:")
            print(status_counts)
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

        print(f"\nTotal number of unique subjects: {total_subjects}")
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
        print(f"\nTotal number of rows in dataset: {total_rows}")
        print("Missing Data Summary:")
        print(missing_data)

    def save_summary(self):
        """
        Save the demographic summary to a CSV file.
        """
        summary_df = pd.DataFrame.from_dict(self.summary, orient='index', columns=['Value'])
        summary_df.to_csv(self.output_file)
        print(f"\nDemographic summary saved to {self.output_file}")

    def main(self):
        """
        Execute the demographic analysis workflow.
        """
        self.load_data()
        self.analyze_demographics_of_actual_participants()
        self.count_complete_participants()
        self.missing_data_summary()
        self.save_summary()


if __name__ == "__main__":
    data_file = "processed_data.csv"
    cross_sectional_file = os.path.join("data", "swan_cross_sectional.tsv")
    analyzer = DemographicAnalyzer(data_file, cross_sectional_file)
    analyzer.main()