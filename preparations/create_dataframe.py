import os
import pandas as pd
import re
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

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

                # Convert specific negative values to NaN - with improved handling
                self.handle_missing_values(df)

                # Append the dataframe to the list
                combined_data.append(df)

        # Combine all dataframes vertically
        if combined_data:
            self.dataframes = pd.concat(combined_data, ignore_index=True)

            # Get initial number of unique subjects
            self.initial_subjects = self.dataframes['SWANID'].nunique()
            self.initial_rows = len(self.dataframes)
            
            # Apply missing value handling one more time to the combined dataset
            self.handle_missing_values(self.dataframes)
        else:
            raise ValueError("No valid .tsv files were processed.")

    def handle_missing_values(self, df):
        """
        Properly handle missing values in the dataframe.
        This handles both string and numeric representations of missing codes.
        
        :param df: DataFrame to process
        """
        # Missing value codes to replace
        missing_codes = [-9, -8, -7, -1]
        string_missing_codes = ["-9", "-8", "-7", "-1", " "]
        
        # First handle string missing codes
        df.replace(string_missing_codes, np.nan, inplace=True)
        
        # Then handle numeric missing codes for each column separately
        for col in df.columns:
            # Try to convert to numeric if possible
            try:
                # Only convert if the column isn't already numeric
                if df[col].dtype == 'object':
                    temp_series = pd.to_numeric(df[col], errors='coerce')
                    
                    # Check if conversion created any NaNs that weren't there before
                    # If so, it means there were non-numeric strings we should preserve
                    newly_missing = temp_series.isna() & df[col].notna()
                    
                    # Replace missing codes with NaN
                    temp_series = temp_series.replace(missing_codes, np.nan)
                    
                    # Put back the original values for strings we want to preserve
                    temp_series.loc[newly_missing] = df.loc[newly_missing, col]
                    
                    # Update the column
                    df[col] = temp_series
                else:
                    # If already numeric, just replace the values
                    df[col] = df[col].replace(missing_codes, np.nan)
            except:
                # If conversion fails, just continue
                continue
        
        # Check for remaining special values, but only print a summary instead of detailed messages
        if len(df) > 0:  # Only check if the dataframe is not empty
            remaining_counts = {}
            for code in missing_codes:
                for col in df.columns:
                    try:
                        count = (df[col] == code).sum()
                        if count > 0:
                            if col not in remaining_counts:
                                remaining_counts[col] = {}
                            remaining_counts[col][code] = count
                    except:
                        continue
            
            # Only print if there are remaining special values
            if remaining_counts:
                print("\nRemaining special values after cleaning:")
                print(f"Found {len(remaining_counts)} columns with special values")
                # Just print the top few columns with the most remaining special values
                top_columns = sorted(remaining_counts.items(), 
                                    key=lambda x: sum(x[1].values()), 
                                    reverse=True)[:3]
                for col, codes in top_columns:
                    total = sum(codes.values())
                    print(f"  Column {col}: {total} total special values")
                
                if len(remaining_counts) > 3:
                    print(f"  ... and {len(remaining_counts) - 3} more columns with special values")
            else:
                print("\nNo remaining special values found after cleaning.")
        
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
        removed_subjects = self.initial_subjects - final_subjects
        final_rows = len(self.dataframes)
        
        print(f"\nSummary of TOTIDE data cleaning:")
        print(f"Initial number of subjects: {self.initial_subjects}")
        print(f"Initial number of rows: {self.initial_rows}")
        print(f"Subjects removed (missing TOTIDE data): {removed_subjects}")
        print(f"Final number of subjects: {final_subjects}")
        print(f"Final number of rows: {final_rows}")
        
        return self.dataframes
    
    def reverse_social_column(self):
        """
        Reverse the SOCIAL column so that higher values indicate worse symptoms, aligning it with other scales.
        The original scale is 1 (all of the time) to 5 (none of the time), so it will be transformed to:
        1 → 5, 2 → 4, 3 → 3, 4 → 2, 5 → 1.
        """
        if 'SOCIAL' in self.dataframes.columns:
            self.dataframes['SOCIAL'] = self.dataframes['SOCIAL'].replace({1: 5, 2: 4, 3: 3, 4: 2, 5: 1})
            print("\n'SOCIAL' column reversed successfully.")
        else:
            print("\n'SOCIAL' column not found in the dataset.")

    def enforce_variable_limits(self):
        """
        Enforces valid value ranges for all variables based on their defined limits.
        Values outside the valid range are capped to the nearest valid value.
        Zero values are preserved (not changed to minimum values).
        Handles both numeric and string values by attempting conversion.
        
        Returns:
            Summary of changes made
        """
        # Define variable limits based on the documentation
        variable_limits = {
            # Cognitive function domain
            'TOTIDE1': {'min': 0, 'max': 12},
            'TOTIDE2': {'min': 0, 'max': 12},
            
            # Symptomatology domain
            'HOTFLAS': {'min': 1, 'max': 5},
            'NUMHOTF': {'min': 1, 'max': 50},
            'BOTHOTF': {'min': 1, 'max': 4},
            'NITESWE': {'min': 1, 'max': 5},
            'NUMNITS': {'min': 1, 'max': 24},
            'BOTNITS': {'min': 1, 'max': 4},
            'COLDSWE': {'min': 1, 'max': 5},
            'BOTCLDS': {'min': 1, 'max': 4},
            'NUMCLDS': {'min': 1, 'max': 15},
            'STIFF': {'min': 1, 'max': 5},
            'IRRITAB': {'min': 1, 'max': 5},
            'MOODCHG': {'min': 1, 'max': 5},
            
            # Social support domain
            'LISTEN': {'min': 1, 'max': 5},
            'TAKETOM': {'min': 1, 'max': 5},
            'HELPSIC': {'min': 1, 'max': 5},
            'CONFIDE': {'min': 1, 'max': 5},
            
            # Emotional impact domain
            'EMOCTDW': {'min': 1, 'max': 2},
            'EMOACCO': {'min': 1, 'max': 2},
            'EMOCARE': {'min': 1, 'max': 2},
            
            # Social health domain
            'INTERFR': {'min': 1, 'max': 5},
            'SOCIAL': {'min': 1, 'max': 5},
            
            # Socioeconomic domain
            'INCOME': {'min': 1, 'max': 4},
            'HOW_HAR': {'min': 1, 'max': 3},
            'BCINCML': {'min': 1, 'max': 2},
            'DEGREE': {'min': 1, 'max': 5},
            
            # Control variables
            'LANGCOG': {'min': 1, 'max': 4},
            'STATUS': {'min': 1, 'max': 8}
        }
        
        # Track changes made
        changes_summary = {}
        
        # Process each variable that has defined limits and exists in the dataframe
        for var, limits in variable_limits.items():
            # Check if the variable exists in our dataframes
            if var in self.dataframes:
                # First, try to convert to numeric, coercing errors to NaN
                try:
                    # Create a copy of the series and convert to numeric
                    numeric_series = pd.to_numeric(self.dataframes[var], errors='coerce')
                    
                    # Check for values outside the limits, only on non-NaN values
                    valid_mask = numeric_series.notna()
                    
                    # Create separate masks for zeros, below min (but not zero), and above max
                    zeros_mask = (numeric_series == 0) & valid_mask
                    below_min = (numeric_series < limits['min']) & (numeric_series != 0) & valid_mask
                    above_max = (numeric_series > limits['max']) & valid_mask
                    
                    # Count values outside limits
                    n_zeros = zeros_mask.sum()
                    n_below = below_min.sum()
                    n_above = above_max.sum()
                    
                    if n_below > 0 or n_above > 0:
                        # Keep track of original values for reporting
                        original_values = self.dataframes.loc[below_min | above_max, var].copy()
                        
                        # Create a temporary series with enforced limits
                        limited_series = numeric_series.copy()
                        limited_series[below_min] = limits['min']  # Set values below min (except 0) to min
                        limited_series[above_max] = limits['max']  # Set values above max to max
                        # Zero values are preserved (not changed)
                        
                        # Only replace values that were successfully converted to numeric
                        self.dataframes.loc[valid_mask, var] = limited_series[valid_mask]
                        
                        # Store summary of changes
                        changes_summary[var] = {
                            'zeros_preserved': n_zeros,
                            'below_min': n_below,
                            'above_max': n_above,
                            'min_limit': limits['min'],
                            'max_limit': limits['max'],
                            'original_values': original_values.tolist() if len(original_values) < 10 else f"{len(original_values)} values outside limits"
                        }
                except Exception as e:
                    print(f"Error processing variable {var}: {e}")
        
        # Print summary report
        if changes_summary:
            print("\nValue Range Enforcement Summary:")
            for var, info in changes_summary.items():
                print(f"\n{var} (range: {info['min_limit']} to {info['max_limit']}):")
                if 'zeros_preserved' in info:
                    print(f"  Zero values preserved: {info['zeros_preserved']}")
                print(f"  Values below minimum: {info['below_min']}")
                print(f"  Values above maximum: {info['above_max']}")
                if isinstance(info['original_values'], list) and len(info['original_values']) > 0:
                    print(f"  Original out-of-range values: {info['original_values']}")
        else:
            print("\nNo values outside valid ranges were found.")
        
        return changes_summary

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
        self.reverse_social_column()
        self.enforce_variable_limits()
        self.process_and_save_data()

if __name__ == "__main__":
    data_folder = 'data'  # Folder containing the .tsv files
    output_file = 'processed_data.csv'  # Output file for the processed data
    cross_sectional_file = 'swan_cross_sectional.tsv'  # Cross-sectional data file
    processor = TSVDataProcessor(data_folder, output_file, cross_sectional_file)
    processor.main()