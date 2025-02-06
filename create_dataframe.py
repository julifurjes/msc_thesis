import os
import pandas as pd
import re

class TSVDataProcessor:
    def __init__(self, data_folder, output_file):
        """
        Initialize the TSVDataProcessor object.

        :param data_folder: The folder where .tsv files are located.
        :param output_file: The name of the file to save the processed data.
        """
        self.data_folder = data_folder
        self.output_file = output_file
        self.dataframes = []

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
            if file_name.endswith('.tsv'):
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

                # Add VISIT column to indicate the timestamp
                df['VISIT'] = visit_number

                # Append the dataframe to the list
                combined_data.append(df)

        # Combine all dataframes vertically
        if combined_data:
            self.dataframes = pd.concat(combined_data, ignore_index=True)
        else:
            raise ValueError("No valid .tsv files were processed.")

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
        self.process_and_save_data()


if __name__ == "__main__":
    data_folder = 'data'  # Folder containing the .tsv files
    output_file = 'processed_combined_data.csv'  # Output file for the processed data
    processor = TSVDataProcessor(data_folder, output_file)
    processor.main()