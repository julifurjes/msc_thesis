import os
import pandas as pd
import re


class TSVDataCombiner:
    def __init__(self, data_folder, output_file):
        """
        Initialize the TSVDataCombiner object.

        :param data_folder: The folder where .tsv files are located.
        :param output_file: The name of the file to save the combined data.
        """
        self.data_folder = data_folder
        self.output_file = output_file
        self.dataframes = []

    def clean_column_names(self, df):
        """
        Remove numbers at the end of column names and visit numbers.

        :param df: DataFrame to clean column names.
        :return: DataFrame with cleaned column names.
        """
        def remove_trailing_numbers(column_name):
            # Remove trailing numbers and whitespace
            return re.sub(r'\s*\d+$', '', column_name)

        cleaned_columns = {col: remove_trailing_numbers(col) for col in df.columns}
        return df.rename(columns=cleaned_columns)

    def load_tsv_files(self):
        """
        Load all .tsv files from the specified folder into a list of DataFrames.
        """
        for file_name in os.listdir(self.data_folder):
            if file_name.endswith('.tsv'):
                file_path = os.path.join(self.data_folder, file_name)
                print(f"Loading file: {file_name}")
                df = pd.read_csv(file_path, sep='\t', low_memory=False)
                # Clean column names before appending
                df = self.clean_column_names(df)
                self.dataframes.append(df)

    def combine_dataframes(self):
        """
        Combine all DataFrames into a single DataFrame, merging duplicate columns.

        :return: A combined DataFrame with merged duplicate columns.
        """
        if not self.dataframes:
            raise ValueError("No dataframes to combine. Please load the .tsv files first.")

        # Concatenate all DataFrames horizontally
        combined_df = pd.concat(self.dataframes, axis=1)

        # Handle duplicate columns by merging them
        combined_df = combined_df.groupby(combined_df.columns, axis=1).first()

        return combined_df

    def save_combined_dataframe(self, combined_df):
        """
        Save the combined DataFrame to the specified output file.

        :param combined_df: The DataFrame to save.
        """
        combined_df.to_csv(self.output_file, index=False)
        print(f"Combined data saved to {self.output_file}")

    def main(self):
        """
        Execute the data loading, combining, and saving process.
        """
        self.load_tsv_files()
        combined_df = self.combine_dataframes()
        print(combined_df.head())
        self.save_combined_dataframe(combined_df)


if __name__ == "__main__":
    data_folder = 'data'
    output_file = 'combined_data.csv'
    combiner = TSVDataCombiner(data_folder, output_file)
    combiner.main()