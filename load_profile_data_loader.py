import numpy as np
import pandas as pd

class LoadProfileDataLoader:

    def __init__(self, csv_file_path: str):
        
        self.file_path = csv_file_path

        
    def read_csv_file(self) -> pd.DataFrame:

        return pd.read_csv(self.file_path, header=0)

    
    def display_data(self, df: pd.DataFrame):

        print(f"\nData successfully loaded from path => {self.file_path}\n")

        print("Info: \n")
        print( df.info() )
        
        print("\nSummary Statistics: \n")
        print( df.describe() )
        print("\n")

    
    def convert_weekday(self, wkd: str) -> str:
        
        if wkd in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            return 'week'
        else:
            return wkd.lower()

    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:

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

        df = self.read_csv_file()
        
        fmt_df = self.preprocess_data(df=df)

        if (from_date is None) and (to_date is None):

            self.display_data(df=fmt_df)

            return fmt_df

        else:

            filtered_fmt_df = fmt_df.loc[from_date : to_date]

            self.display_data(df=filtered_fmt_df)

            return filtered_fmt_df

