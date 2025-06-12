import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_date_info():
    """
    Generates a list of date information including row number, date, hour, and weekday
    for the period of November 23rd to November 30th, 2024.
    """
    start_date = datetime.date(2024, 11, 23)
    end_date = datetime.date(2024, 11, 30)

    # List to store all generated date data rows
    date_info_rows = []
    row_number = 0

    # Calculate the number of days in the range
    delta = end_date - start_date
    num_days = delta.days + 1 # +1 to include the end_date itself

    # Iterate through each day in the specified range
    for i in range(num_days):
        current_date = start_date + datetime.timedelta(days=i)
        # Format the date as 'YYYY-MM-DD'
        formatted_date = current_date.strftime('%Y-%m-%d')
        # Get the full weekday name
        weekday_name = current_date.strftime('%A')

        # Iterate through each hour of the day (0 to 23)
        for hour in range(24):
            # Create a dictionary for the current row's date data
            row_data = {
                "Row Number": row_number,
                "Date": formatted_date,
                "Hour": hour,
                "Weekday": weekday_name
            }
            date_info_rows.append(row_data)
            row_number += 1

    # Convert the list of dictionaries to a pandas DataFrame
    date_df = pd.DataFrame(date_info_rows)
    return date_df

def join_data_with_csv(date_dataframe, csv_file_path):
    """
    Reads a CSV file and joins its data with the provided date_dataframe
    on the 'Step' column from CSV and 'Row Number' from date_dataframe.

    Args:
        date_dataframe (pd.DataFrame): DataFrame containing date information.
        csv_file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    try:
        # Read the CSV file into a pandas DataFrame
        csv_df = pd.read_csv(csv_file_path)

        # Ensure the 'Step' column in CSV is of appropriate type for merging, if needed.
        # For this scenario, assuming 'Step' is numerical and matches 'Row Number'.
        if 'Step' not in csv_df.columns:
            print(f"Error: The CSV file '{csv_file_path}' does not contain a 'Step' column.")
            return None

        # Perform the merge operation
        # Using a left merge to keep all date info rows and add matching CSV data
        merged_df = pd.merge(date_dataframe, csv_df, left_on='Row Number', right_on='Step', how='left')

        # Drop the redundant 'Step' column from the merged DataFrame, as 'Row Number' covers it
        merged_df = merged_df.drop(columns=['Step'])

        return merged_df

    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred during CSV processing or merging: {e}")
        return None

def plot_bess_soc(dataframe):
    """
    Plots the BESS SoC over time.

    Args:
        dataframe (pd.DataFrame): The merged DataFrame containing 'Date', 'Hour', and 'bess_soc'.
    """
    if 'Date' not in dataframe.columns or 'Hour' not in dataframe.columns or 'bess_soc' not in dataframe.columns:
        print("Error: DataFrame must contain 'Date', 'Hour', and 'bess_soc' columns for plotting.")
        return

    # Create a combined datetime column for the x-axis
    # Convert 'Hour' to string for proper concatenation as a time string (e.g., '00', '01', ..., '23')
    dataframe['Time'] = dataframe['Date'] + ' ' + dataframe['Hour'].astype(str).str.zfill(2) + ':00'

    # Convert the 'Time' column to datetime objects for proper plotting order
    dataframe['Time'] = pd.to_datetime(dataframe['Time'])

    # Set up the plot style
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6)) # Adjust figure size as needed

    # Create the line plot
    sns.lineplot(data=dataframe, x='Time', y='bess_soc', marker='o', linestyle='-')

    # Set titles and labels
    plt.title("RL Agent - BESS SoC Management", fontsize=16)
    plt.xlabel("Time (<Year>-<Month>-<Day> <Hour>)", fontsize=12)
    plt.ylabel("BESS SoC", fontsize=12)

    # Improve x-axis tick readability
    plt.xticks(rotation=45, ha='right') # Rotate labels for better fit
    plt.grid(True)
    plt.tight_layout() # Adjust layout to prevent labels from overlapping

    # Save the plot
    plot_filename = "bess_soc_management_plot.png"
    plt.savefig(plot_filename)
    print(f"\nPlot saved as '{plot_filename}'")
    # plt.show() # Uncomment this line if you want to display the plot immediately (requires a GUI environment)

if __name__ == "__main__":
    # Generate the base date information DataFrame
    generated_date_df = generate_date_info()

    # Define the path to your uploaded CSV file
    csv_file_name = "Jun12_19-45-22_Arnos-MacBook-Pro.localDoubleDQNAgent-TestEnv_Microgrid Environment State_bess_soc.csv"

    # Join the generated date data with the CSV data
    final_merged_df = join_data_with_csv(generated_date_df, csv_file_name)

    if final_merged_df is not None:
        # Print the merged DataFrame (optional, can be commented out if only plot is needed)
        pd.set_option('display.max_columns', None)
        print("\nMerged Data (first 5 rows):")
        print(final_merged_df.head().to_string()) # Print only head for brevity

        # Plot the BESS SoC data
        plot_bess_soc(final_merged_df)
    else:
        print("Data merging failed, cannot generate plot.")
