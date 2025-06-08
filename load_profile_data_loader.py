import numpy as np
import pandas as pd

class LoadProfileDataLoader:
    """
    A utility class for loading and preprocessing microgrid load profile data from CSV files.

    This class handles reading CSV data, performing necessary transformations like
    datetime conversion, feature engineering (e.g., one-hot encoding for time-based features),
    and filtering data based on date ranges.
    """

    def __init__(self, csv_file_path: str):
        """
        Initializes the LoadProfileDataLoader.

        Args:
            csv_file_path (str): The path to the CSV file containing the load profile data.
        """

        self.file_path = csv_file_path

        
    def read_csv_file(self) -> pd.DataFrame:
        """
        Reads the CSV file specified during initialization.

        Returns:
            pd.DataFrame: A Pandas DataFrame containing the raw data from the CSV file.
        """

        return pd.read_csv(self.file_path, header=0)

    
    def display_data(self, df: pd.DataFrame):
        """
        Prints information and summary statistics for the given DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to display information about.
        """

        print(f"\nData successfully loaded from path => {self.file_path}\n")

        print("Info: \n")
        print( df.info() )
        
        print("\nSummary Statistics: \n")
        print( df.describe() )
        print("\n")

    
    def convert_weekday(self, wkd: str) -> str:
        """
        Converts a full weekday name (e.g., 'Monday') to a simplified category ('week', 'saturday', 'sunday').

        Args:
            wkd (str): The full weekday name.

        Returns:
            str: The simplified weekday category.
        """

        if wkd in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            return 'week'
        else:
            return wkd.lower()

    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the raw load profile DataFrame.

        This includes:
        - Converting entry time to datetime objects.
        - Extracting the hour of the day.
        - One-hot encoding weekday and time-of-use (TOU) timeslots.
        - Converting the hour into cyclical sine and cosine features.
        - Setting the timestamp as the index.
        - Dropping unnecessary original columns.
        - Selecting and ordering the final set of features.

        Args:
            df (pd.DataFrame): The raw DataFrame to preprocess.

        Returns:
            pd.DataFrame: The preprocessed DataFrame with engineered features.
        """

        df['timestamp'] = pd.to_datetime(df['entry_time'], format="mixed")
        df['ts_hour'] = df['timestamp'].dt.hour

        # One-hot encode the Weekday field
        df['weekday_fmt'] = df['weekday'].map(lambda wkd: self.convert_weekday(wkd=wkd))
        df['day_week'] = df['weekday_fmt'].map(lambda wkd: 1 if wkd == 'week' else 0)
        df['day_saturday'] = df['weekday_fmt'].map(lambda wkd: 1 if wkd == 'saturday' else 0)
        df['day_sunday'] = df['weekday_fmt'].map(lambda wkd: 1 if wkd == 'sunday' else 0)

        # One-hot encode the TOU Timeslot field
        df['tou_offpeak'] = df['tou_time_slot'].map(lambda tou: 1 if tou == 'o' else 0)
        df['tou_standard'] = df['tou_time_slot'].map(lambda tou: 1 if tou == 's' else 0)
        df['tou_peak'] = df['tou_time_slot'].map(lambda tou: 1 if tou == 'p' else 0)

        # Convert hour field to unit circle coordinates
        df['ts_hour_sin'] = np.sin( df['ts_hour'] )
        df['ts_hour_cos'] = np.cos( df['ts_hour'] )

        df.set_index('timestamp', inplace=True)

        df.drop(['entry_time', 'ts_hour', 'weekday', 'weekday_fmt', 'tou_time_slot'], axis=1, inplace=True)

        return df[['ts_hour_sin', 'ts_hour_cos', 'tou_offpeak', 'tou_standard', 'tou_peak', 'day_week', 'day_saturday', 'day_sunday', 'site_load_energy', 'solar_prod_energy', 'solar_ctlr_setpoint', 'grid_import_energy']]


    def load_data(self, from_date: str = None, to_date: str = None) -> pd.DataFrame:
        """
        Loads, preprocesses, and optionally filters the load profile data by date.

        Args:
            from_date (str, optional): The start date for filtering (YYYY-MM-DD). Defaults to None.
            to_date (str, optional): The end date for filtering (YYYY-MM-DD). Defaults to None.

        Returns:
            pd.DataFrame: The processed (and potentially filtered) DataFrame.
        """

        df = self.read_csv_file()
        
        fmt_df = self.preprocess_data(df=df)

        if (from_date is None) and (to_date is None):

            self.display_data(df=fmt_df)

            return fmt_df

        else:

            filtered_fmt_df = fmt_df.loc[from_date : to_date]

            self.display_data(df=filtered_fmt_df)

            return filtered_fmt_df

