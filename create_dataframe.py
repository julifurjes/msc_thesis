import os
import pandas as pd
import re
import numpy as np

class TSVDataProcessor:
    def __init__(self, data_folder, output_file, cross_sectional_file):
        """
        Initialize the TSVDataProcessor object.

        :param data_folder: The folder where .tsv files are located.
        :param output_file: The name of the file to save the processed data.
        """
        self.data_folder = data_folder
        self.output_file = output_file
        self.dataframes = []
        self.cross_sectional_file = cross_sectional_file

    def clean_column_names(self, df, visit_number):
        """
        Remove the visit number from column names while keeping other numbers intact.

        :param df: DataFrame to clean column names.
        :param visit_number: The visit number to remove from column names.
        :return: DataFrame with cleaned column names.
        """
        def remove_visit_number(column_name):
            # Remove the specific visit number at the end, preceded by an optional space
            return re.sub(fr'(\D*)\s*{visit_number}$', r'\1', column_name)

        cleaned_columns = {col: remove_visit_number(col) for col in df.columns}
        return df.rename(columns=cleaned_columns)

    def load_and_combine_tsv_files(self):
        """
        Load all .tsv files from the specified folder, clean column names, and combine them into one DataFrame.
        """
        combined_data = []

        for file_name in os.listdir(self.data_folder):
            if file_name.endswith('.tsv') and 'swan_cross_sectional' not in file_name:
                # Extract the visit number from the filename (assumes format includes 'timestampN')
                match = re.search(r'timestamp_(\d+)', file_name)
                if match:
                    visit_number = int(match.group(1))
                    print(f"Processing file: {file_name} with visit number: {visit_number}")
                else:
                    print(f"Skipping file: {file_name} (no visit number detected)")
                    continue

                # Load the TSV file
                file_path = os.path.join(self.data_folder, file_name)
                df = pd.read_csv(file_path, sep='\t', low_memory=False)

                # Clean column names to remove visit number
                df = self.clean_column_names(df, visit_number)

                # Print current visit number and unique values in the VISIT column
                print(f"Unique values in VISIT column: {df['VISIT'].unique()}")
                print("Visit number: ", visit_number)

                # Convert specific negative values to NaN
                special_na_values = ["-9", "-7", "-8", " "]
                for sc in special_na_values:
                    df.replace(sc, np.nan, inplace=True)

                # Append the dataframe to the list
                combined_data.append(df)

        # Combine all dataframes vertically
        if combined_data:
            self.dataframes = pd.concat(combined_data, ignore_index=True)
        else:
            raise ValueError("No valid .tsv files were processed.")
        
    def merge_cross_sectional_data(self):
        """
        Load the cross-sectional data and merge the DEGREE column with the longitudinal dataset.
        """
        # Load the cross-sectional data
        cross_sectional_path = os.path.join(self.data_folder, self.cross_sectional_file)
        cross_sectional_df = pd.read_csv(cross_sectional_path, sep='\t', low_memory=False)

        # Merge only the DEGREE column with the main dataset
        self.dataframes = self.dataframes.merge(
            cross_sectional_df[['SWANID', 'DEGREE']], 
            on='SWANID', 
            how='left'
        )

        # Calculate and print the number of SWANIDs with missing DEGREE values
        missing_degree_count = self.dataframes[
            self.dataframes['DEGREE'].isna()
        ]['SWANID'].nunique()
        
        print(f"\nSummary of missing DEGREE values:")
        print(f"Number of unique SWANIDs with missing DEGREE: {missing_degree_count}")

    def remove_missing_totide_subjects(self):
        """
        Remove subjects who have both TOTIDE1 and TOTIDE2 missing for all their visits.
        """
        # Get initial number of unique subjects
        initial_subjects = self.dataframes['SWANID'].nunique()
        
        # For each subject, check if they have any non-missing TOTIDE1 or TOTIDE2 values
        totide_counts = self.dataframes.groupby('SWANID').agg({
            'TOTIDE1': lambda x: x.notna().sum(),
            'TOTIDE2': lambda x: x.notna().sum()
        })
        
        # Keep subjects who have at least two non-missing values in both TOTIDE1 and TOTIDE2
        subjects_to_keep = totide_counts[
            (totide_counts['TOTIDE1'] >= 2) & (totide_counts['TOTIDE2'] >= 2)
        ].index
                
        # Filter the dataframe to keep only these subjects
        self.dataframes = self.dataframes[
            self.dataframes['SWANID'].isin(subjects_to_keep)
        ]
        
        # Calculate and print summary statistics
        final_subjects = self.dataframes['SWANID'].nunique()
        removed_subjects = initial_subjects - final_subjects
        
        print(f"\nSummary of TOTIDE data cleaning:")
        print(f"Initial number of subjects: {initial_subjects}")
        print(f"Subjects removed (missing TOTIDE data): {removed_subjects}")
        print(f"Final number of subjects: {final_subjects}")
        
        return self.dataframes

    def process_and_save_data(self):
        """
        Save the combined DataFrame to the output file.
        """
        if self.dataframes.empty:
            raise ValueError("No data to process. Please load and combine the .tsv files first.")

        # Save the combined DataFrame to the output file
        self.dataframes.to_csv(self.output_file, index=False)
        print(f"Processed data saved to {self.output_file}")

    def main(self):
        """
        Execute the data loading, cleaning, and saving process.
        """
        self.load_and_combine_tsv_files()
        print(self.dataframes.head())
        print(self.dataframes['VISIT'].unique())
        self.merge_cross_sectional_data()
        self.remove_missing_totide_subjects()
        self.process_and_save_data()

if __name__ == "__main__":
    data_folder = 'data'  # Folder containing the .tsv files
    output_file = 'processed_data.csv'  # Output file for the processed data
    cross_sectional_file = 'swan_cross_sectional.tsv'  # Cross-sectional data file
    processor = TSVDataProcessor(data_folder, output_file, cross_sectional_file)
    processor.main()